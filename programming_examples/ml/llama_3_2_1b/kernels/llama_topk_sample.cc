//===- llama_topk_sample.cc ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Samples the top-k set llama_lmhead_topk.cc merges, then makes the next
// token's embed seed from the winner's row, which the set carries.
// llama_oracle.py is the oracle.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

#ifndef LLAMA_TOPK_D
#define LLAMA_TOPK_D 2048
#endif
#ifndef LLAMA_TOPK_NTILE
#define LLAMA_TOPK_NTILE 4
#endif
#ifndef LLAMA_TOPK_KSET
#define LLAMA_TOPK_KSET 8
#endif

static constexpr int kD = LLAMA_TOPK_D;
static constexpr int kNTile = LLAMA_TOPK_NTILE;
static constexpr int kKset = LLAMA_TOPK_KSET;

// xoshiro128++, as llama_oracle draws it.
struct Xoshiro128 {
  uint32_t s[4];
};
static inline uint32_t rotl(uint32_t x, int k) {
  uint64_t y = ((uint64_t)x << 32) | (uint64_t)x;
  return (uint32_t)(y >> (32 - k));
}
static inline uint32_t xoshiro_next(Xoshiro128 &g) {
  uint32_t result = rotl(g.s[0] + g.s[3], 7) + g.s[0];
  uint32_t t = g.s[1] << 9;
  g.s[2] ^= g.s[0];
  g.s[3] ^= g.s[1];
  g.s[1] ^= g.s[2];
  g.s[0] ^= g.s[3];
  g.s[2] ^= t;
  g.s[3] = rotl(g.s[3], 11);
  return result;
}
static inline float xoshiro_uniform(Xoshiro128 &g) {
  uint32_t r = xoshiro_next(g);
  return (float)(r >> 8) * (1.0f / 16777216.0f);
}

// A token's layer-0 residual row from its embed row (row, embed_sc): the row
// in fp32, row * embed_sc, which the first norm reads and the residual add
// adds to.
static void residual_seed(const int8_t *restrict row, float embed_sc,
                          int8_t *restrict seed) {
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  ::aie::vector<float, 16> sc_v = ::aie::broadcast<float, 16>(embed_sc);
  float *x = reinterpret_cast<float *>(seed);
  for (int c = 0; c < kD / 16; c++)
    ::aie::store_v(x + c * 16,
                   vec_mul_rne(::aie::to_float<float>(
                                   ::aie::load_unaligned_v<16>(row + c * 16)),
                               sc_v));
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
}

extern "C" {

// Samples the set: the token id, and the winner's embed row and scale.
// params is [temperature bits | top_k | xoshiro state x4]; a sampled pick
// advances the state in place. Sampling draws from the set's best top_k (by
// logit, then lower gidx; 0 or past the set: all of it). Greedy
// (temperature <= 0) is the argmax whatever top_k is. token gets the id,
// record the winner's embed row, then its embed_sc.
void llama_topk_finalize(float *restrict set_logit, int32_t *restrict set_gidx,
                         float *restrict set_scale, int8_t *restrict set_row,
                         int32_t *restrict set_len, uint32_t *restrict params,
                         int32_t *restrict token, int8_t *restrict record) {
  const int len = *set_len;
  float temperature;
  uint32_t tbits = params[0];
  __builtin_memcpy(&temperature, &tbits, 4);

  int winner;
  if (temperature <= 0.0f) {
    // Greedy (see llama_oracle.topk_sample_reference's matching tie-break
    // rule).
    float wv = set_logit[0];
    for (int i = 1; i < len; i++)
      if (set_logit[i] > wv)
        wv = set_logit[i];
    winner = -1;
    int32_t best_g = 0;
    for (int i = 0; i < len; i++) {
      if (set_logit[i] == wv && (winner < 0 || set_gidx[i] < best_g)) {
        winner = i;
        best_g = set_gidx[i];
      }
    }
  } else {
    int32_t top_k = (int32_t)params[1];
    if (top_k < 1 || top_k > len)
      top_k = len;
    bool keep[kKset];
    for (int i = 0; i < len; i++) {
      int better = 0;
      for (int j = 0; j < len; j++)
        better += set_logit[j] > set_logit[i] ||
                  (set_logit[j] == set_logit[i] && set_gidx[j] < set_gidx[i]);
      keep[i] = better < top_k;
    }
    const float inv_temp = recip_rne(temperature);
    float max_z = set_logit[0] * inv_temp;
    for (int i = 1; i < len; i++) {
      float z = set_logit[i] * inv_temp;
      if (z > max_z)
        max_z = z;
    }
    // Each member's weight e^(z - max_z), 16 at a time.
    alignas(64) float e[(kKset + 15) / 16 * 16] = {};
    for (int i = 0; i < len; i++)
      e[i] = set_logit[i] * inv_temp - max_z;
    const ::aie::saturation_mode prev_sat =
        ::aie::swap_saturation(::aie::saturation_mode::saturate);
    const ::aie::rounding_mode prev_rnd =
        ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    for (int i = 0; i < len; i += 16)
      ::aie::store_v(e + i, vec_exp_nonpos(::aie::load_v<16>(e + i)));
    ::aie::set_rounding(prev_rnd);
    ::aie::set_saturation(prev_sat);

    // Sum and walk the survivors in ascending gidx, the oracle's order: fp32
    // adds don't associate, and an ulp moves a pick near a CDF boundary.
    float sum_exp = 0.0f;
    int prev_g = -1;
    for (int step = 0; step < top_k; step++) {
      int si = -1;
      int32_t sg = 0;
      for (int i = 0; i < len; i++) {
        if (keep[i] && set_gidx[i] > prev_g && (si < 0 || set_gidx[i] < sg)) {
          si = i;
          sg = set_gidx[i];
        }
      }
      prev_g = sg;
      sum_exp += e[si];
    }

    Xoshiro128 g;
    for (int i = 0; i < 4; i++)
      g.s[i] = params[2 + i];
    float u = xoshiro_uniform(g) * sum_exp;
    for (int i = 0; i < 4; i++)
      params[2 + i] = g.s[i];

    float cum = 0.0f;
    winner = -1;
    prev_g = -1;
    for (int step = 0; step < top_k; step++) {
      // The next survivor by gidx.
      int si = -1;
      int32_t sg = 0;
      for (int i = 0; i < len; i++) {
        if (keep[i] && set_gidx[i] > prev_g && (si < 0 || set_gidx[i] < sg)) {
          si = i;
          sg = set_gidx[i];
        }
      }
      prev_g = sg;
      cum += e[si];
      if (cum > u) {
        winner = si;
        break;
      }
      winner = si; // fallback to last if u falls off the end (fp roundoff)
    }
  }

  token[0] = set_gidx[winner];

  copy_words<kD>(record, set_row + winner * kD);
  __builtin_memcpy(record + kD, &set_scale[winner], 4);
}

// llama_topk_finalize with both outputs in one buffer, out[kD + 12]:
//   [0..kD) int8 row | [kD..kD+4) f32 scale | [kD+4..kD+8) i32 id | pad
void llama_topk_finalize_packed(float *restrict set_logit,
                                int32_t *restrict set_gidx,
                                float *restrict set_scale,
                                int8_t *restrict set_row,
                                int32_t *restrict set_len,
                                uint32_t *restrict params,
                                int8_t *restrict out) {
  int32_t token;
  llama_topk_finalize(set_logit, set_gidx, set_scale, set_row, set_len, params,
                      &token, out);
  __builtin_memcpy(out + kD + 4, &token, 4);
  int32_t zero = 0;
  __builtin_memcpy(out + kD + 8, &zero, 4);
}

// The generation's xoshiro state, carried in the stage between dispatches:
// from the head of a slot into params[2..6) at a dispatch's start, and back
// into a row at its end.
void llama_rng_take(int8_t *restrict slot, uint32_t *restrict params) {
  copy_words<16>(params + 2, slot);
}

void llama_rng_put(uint32_t *restrict params, int8_t *restrict row) {
  copy_words<16>(row, params + 2);
  zero_words<4 * kD - 16>(row + 16);
}

// The next token's layer-0 residual row from a packed record.
void llama_seed_from_record(int8_t *restrict record, int8_t *restrict seed) {
  float embed_sc;
  __builtin_memcpy(&embed_sc, record + kD, 4);
  residual_seed(record, embed_sc, seed);
}

// The embed seed of row `row` of one lm_head weight slot, [kNTile int8 rows
// | 8 i32 bias | 8 f32 embed_sc]: a prompt token's layer-0 input.
void llama_embed_seed(int8_t *restrict slot, int8_t *restrict seed,
                      int32_t row) {
  float embed_sc;
  __builtin_memcpy(&embed_sc, slot + kNTile * kD + 32 + row * 4, 4);
  residual_seed(slot + row * kD, embed_sc, seed);
}

} // extern "C"

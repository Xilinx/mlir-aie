//===- llama_topk_sample.cc ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// One-stream sampler + embed-gather. Streaming the lm_head's 262 MB tied
// embed table TWICE per token (logits, then gather the winner's embed)
// wastes ~25% of decode's bandwidth; this streams it ONCE, keeping a
// resident top-k of {logit, global index, embed_sc, embed ROW} -- the
// winner's embedding is already resident when sampling finishes. Clean
// top-k renorm (can't re-touch all V logits); numpy_topk_sample.py is
// the oracle. top_k==0 unsupported (see README). Peano AIE2P: sw_recip
// not `1.0f/x`; rotl via concat-shift; no fp32 array live across loops.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "exp_lut.h"
#include "llama_vec_fp.h"

#ifndef LLAMA_TOPK_D
#define LLAMA_TOPK_D 2048
#endif
#ifndef LLAMA_TOPK_NTILE
#define LLAMA_TOPK_NTILE 4
#endif
#ifndef LLAMA_TOPK_KSET
#define LLAMA_TOPK_KSET 64
#endif
#ifndef LLAMA_TOPK_EXP_QUANT_SCALE
#define LLAMA_TOPK_EXP_QUANT_SCALE 0.05f
#endif

static constexpr int kD = LLAMA_TOPK_D;
static constexpr int kNTile = LLAMA_TOPK_NTILE;
static constexpr int kKset = LLAMA_TOPK_KSET;
static constexpr float kExpQuantScale = LLAMA_TOPK_EXP_QUANT_SCALE;
static constexpr float kInvExpQuantScale = 1.0f / LLAMA_TOPK_EXP_QUANT_SCALE;
static constexpr int kVec = 64;

static inline int32_t quant_shifted(float shifted) {
  float v = shifted * kInvExpQuantScale;
  int32_t q = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (q > 0)
    q = 0;
  if (q < -128)
    q = -128;
  return q;
}

static inline float lookup_exp(int32_t q) {
  uint32_t raw = llama_exp_lut_raw[q + 128];
  float v = __builtin_bit_cast(float, raw);
  return v;
}

static inline float sw_recip(float a) {
  int32_t bits = __builtin_bit_cast(int32_t, a);
  bits = (int32_t)0x7EF477D5 - bits;
  float x = __builtin_bit_cast(float, bits);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

// --- xoshiro128++ (must match llama_sample.cc / numpy_topk_sample.py) ---
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
static inline void xoshiro_seed(Xoshiro128 &g, uint32_t seed) {
  uint64_t x = ((uint64_t)seed) * 0x9E3779B97F4A7C15ULL + 1ULL;
  for (int i = 0; i < 4; i++) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    g.s[i] = (uint32_t)(x ^ (x >> 32));
  }
}
static inline float xoshiro_uniform(Xoshiro128 &g) {
  uint32_t r = xoshiro_next(g);
  return (float)(r >> 8) * (1.0f / 16777216.0f);
}

// The embed seed of one embed row: the row requantized to its own absmax,
// out[kD + 8] = [int8 row | f32 scale | 0]. embed_sc cancels
// (test_chain_persist_grow_mh.embed_seed_ref).
static void embed_seed(const int8_t *restrict row, float embed_sc,
                       int8_t *restrict seed) {
  ::aie::vector<int8, kVec> hi = ::aie::load_unaligned_v<kVec>(row);
  ::aie::vector<int8, kVec> lo = hi;
  for (int c = 1; c < kD / kVec; c++) {
    ::aie::vector<int8, kVec> v = ::aie::load_unaligned_v<kVec>(row + c * kVec);
    hi = ::aie::max(hi, v);
    lo = ::aie::min(lo, v);
  }
  int32_t amax = ::aie::reduce_max(hi);
  if (-(int32_t)::aie::reduce_min(lo) > amax)
    amax = -(int32_t)::aie::reduce_min(lo);
  if (amax < 1)
    amax = 1;
  float inv = sw_recip((float)amax) * 127.0f; // 127/amax
  // (float)row * inv and round_to_i8 on 16 lanes: vec_mul_rne is the soft
  // `*`, and vec_round_to_i8 rounds as round_to_i8 does.
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  ::aie::vector<float, 16> inv_v = ::aie::broadcast<float, 16>(inv);
  for (int c = 0; c < kD / 16; c++)
    ::aie::store_v(
        seed + c * 16,
        vec_round_to_i8(vec_mul_rne(
            ::aie::to_float<float>(::aie::load_unaligned_v<16>(row + c * 16)),
            inv_v)));
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  float scale = (float)amax * embed_sc * (1.0f / 127.0f);
  __builtin_memcpy(seed + kD, &scale, 4);
  int32_t zero = 0;
  __builtin_memcpy(seed + kD + 4, &zero, 4);
}

extern "C" {

// Finalize: sample over the resident set -> token id + next-token embed seed.
//   params : [temperature bits f32 | top_k i32 | seed u32]
// Sampling draws from the set's best top_k members (by logit, then lower
// gidx); top_k 0, or past the set, means the whole set (hosts reject values
// past KSET). Greedy (temperature <= 0) is the argmax whatever top_k is.
//   token  : out int32[1] (global token id)
//   seed   : out int8[kD + 8] (requant'd winner embed row + fp32 scale tail)
void llama_topk_finalize(float *restrict set_logit, int32_t *restrict set_gidx,
                         float *restrict set_scale, int8_t *restrict set_row,
                         int32_t *restrict set_len, uint32_t *restrict params,
                         int32_t *restrict token, int8_t *restrict seed) {
  const int len = *set_len;
  float temperature;
  uint32_t tbits = params[0];
  __builtin_memcpy(&temperature, &tbits, 4);
  const uint32_t seed_rng = params[2];

  int winner;
  if (temperature <= 0.0f) {
    // Greedy (see numpy_topk_sample.py's matching tie-break rule).
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
    const float inv_temp = sw_recip(temperature);
    float max_z = set_logit[0] * inv_temp;
    for (int i = 1; i < len; i++) {
      float z = set_logit[i] * inv_temp;
      if (z > max_z)
        max_z = z;
    }

    // sum_exp AND the inverse-CDF walk must both traverse the survivors in
    // ascending global-index order: fp32 add is not associative, so the
    // accumulation order must match the oracle (which sums in gidx order) or
    // sum_exp differs by an ULP and the pick flips near a CDF boundary.
    // selection over indices (len <= kKset is small, no extra storage).
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
      int32_t q = quant_shifted(set_logit[si] * inv_temp - max_z);
      sum_exp += lookup_exp(q);
    }

    Xoshiro128 g;
    xoshiro_seed(g, seed_rng);
    float u = xoshiro_uniform(g) * sum_exp;

    float cum = 0.0f;
    winner = -1;
    prev_g = -1;
    for (int step = 0; step < top_k; step++) {
      // next smallest gidx strictly greater than prev_g
      int si = -1;
      int32_t sg = 0;
      for (int i = 0; i < len; i++) {
        if (keep[i] && set_gidx[i] > prev_g && (si < 0 || set_gidx[i] < sg)) {
          si = i;
          sg = set_gidx[i];
        }
      }
      prev_g = sg;
      int32_t q = quant_shifted(set_logit[si] * inv_temp - max_z);
      cum += lookup_exp(q);
      if (cum > u) {
        winner = si;
        break;
      }
      winner = si; // fallback to last if u falls off the end (fp roundoff)
    }
  }

  token[0] = set_gidx[winner];

  embed_seed(set_row + winner * kD, set_scale[winner], seed);
}

// Packed-output finalize: identical sampling, but writes BOTH the next-token
// embed seed AND the token id into ONE buffer out[kD + 12]:
//   [0..kD)        int8 embed row (requant'd)
//   [kD..kD+4)     fp32 per-token scale
//   [kD+4..kD+8)   int32 token id
//   [kD+8..kD+12)  pad
// This keeps the chain's runtime arg count at 5 (DefaultNPURuntime.run_test
// segfaults at ~6 tensor args -- the IRON-constraints memo). It is also the
// natural persistent-loop shape: the seed feeds back as the next layer-0 input
// and the token rides along for the host to read (or to drive embed-gather).
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
}

// The next token's layer-0 input from a packed record: the embed row and its
// scale, then zeros to the activation's kD + 16 bytes.
void llama_seed_from_record(int8_t *restrict record, int8_t *restrict seed) {
  copy_words<kD + 4>(seed, record);
  zero_words<12>(seed + kD + 4);
}

// The embed seed of row `row` of one lm_head weight slot, [64 B prefix |
// kNTile int8 rows | kNTile i32 bias | kNTile f32 embed_sc]: a prompt
// token's layer-0 input.
void llama_embed_seed(int8_t *restrict slot, int8_t *restrict seed,
                      int32_t row) {
  constexpr int kPrefix = 64;
  float embed_sc;
  __builtin_memcpy(&embed_sc,
                   slot + kPrefix + kNTile * kD + kNTile * 4 + row * 4, 4);
  embed_seed(slot + kPrefix + row * kD, embed_sc, seed);
}

} // extern "C"

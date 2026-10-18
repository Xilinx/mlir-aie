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
#include <string.h>

#include "build/exp_lut.h"

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

// Exact byte copy of one kD-element embed row, kVec lanes at a time (no
// float rounding -- just a vectorized memcpy). kD is a multiple of kVec
// for every shape this kernel is built at (D=256/2048, kVec=64), but
// neither pointer is guaranteed 64-byte aligned -- src is offset by
// kRowsOff (32 B) from the streamed slot's base, and dst is offset by
// a multiple of kD from set_row's base -- so both sides use unaligned
// load/store.
static inline void vec_copy_row(int8_t *restrict dst,
                                const int8_t *restrict src) {
  static_assert(kD % kVec == 0, "LLAMA_TOPK_D must be a multiple of 64");
  for (int c = 0; c < kD / kVec; c++)
    ::aie::store_unaligned_v(dst + c * kVec,
                             ::aie::load_unaligned_v<kVec>(src + c * kVec));
}

// Packed input slot per tile (mirrors the real lm_head weight slot shape so the
// fused version reuses the same stream):
//   [kNTile fp32 logits | kNTile fp32 embed_sc | kNTile*kD int8 embed rows]
static constexpr int kLogitsOff = 0;
static constexpr int kScalesOff = kNTile * 4;
static constexpr int kRowsOff = kNTile * 4 + kNTile * 4;

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
  float v;
  memcpy(&v, &raw, 4);
  return v;
}

static inline float sw_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits;
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

static inline int8_t round_to_i8(float v) {
  float r = v >= 0.0f ? (v + 0.5f) : (v - 0.5f);
  int32_t i = (int32_t)r;
  if (i > 127)
    i = 127;
  if (i < -128)
    i = -128;
  return (int8_t)i;
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

extern "C" {

// Insert one streamed slot of kNTile rows into the resident top-k set
// (set_logit/gidx/scale/row, capacity kKset, set_len tracks occupancy).
// Min-eviction unsorted set: append until full, then replace the current
// min slot if a strictly-larger logit arrives -- for DISTINCT fp32
// logits (real GEMM output, effectively always) this yields exactly the
// kKset largest, matching the oracle's first-seen tie-break.
void llama_topk_insert(int8_t *restrict slot, float *restrict set_logit,
                       int32_t *restrict set_gidx, float *restrict set_scale,
                       int8_t *restrict set_row, int32_t *restrict set_len,
                       int32_t tile_idx) {
  const float *logits = reinterpret_cast<const float *>(slot + kLogitsOff);
  const float *scales = reinterpret_cast<const float *>(slot + kScalesOff);
  const int8_t *rows = slot + kRowsOff;
  const int32_t base = tile_idx * kNTile;

  for (int j = 0; j < kNTile; j++) {
    const float lg = logits[j];
    const int32_t gidx = base + j;
    const float sc = scales[j];
    const int8_t *row = rows + j * kD;

    int len = *set_len;
    if (len < kKset) {
      set_logit[len] = lg;
      set_gidx[len] = gidx;
      set_scale[len] = sc;
      vec_copy_row(set_row + len * kD, row);
      *set_len = len + 1;
      continue;
    }
    // find current-min slot
    int mi = 0;
    float mv = set_logit[0];
    for (int i = 1; i < kKset; i++) {
      if (set_logit[i] < mv) {
        mv = set_logit[i];
        mi = i;
      }
    }
    if (lg > mv) {
      set_logit[mi] = lg;
      set_gidx[mi] = gidx;
      set_scale[mi] = sc;
      vec_copy_row(set_row + mi * kD, row);
    }
  }
}

// Finalize: sample over the resident set -> token id + next-token embed seed.
//   params : [temperature bits f32 | top_k i32 | seed u32]
//   token  : out int32[1] (global token id)
//   seed   : out int8[kD + 8] (requant'd winner embed row + fp32 scale tail)
void llama_topk_finalize(float *restrict set_logit, int32_t *restrict set_gidx,
                         float *restrict set_scale, int8_t *restrict set_row,
                         int32_t *restrict set_len, uint32_t *restrict params,
                         int32_t *restrict token, int8_t *restrict seed) {
  const int len = *set_len;
  float temperature;
  uint32_t tbits = params[0];
  memcpy(&temperature, &tbits, 4);
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
    for (int step = 0; step < len; step++) {
      int si = -1;
      int32_t sg = 0;
      for (int i = 0; i < len; i++) {
        if (set_gidx[i] > prev_g && (si < 0 || set_gidx[i] < sg)) {
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
    for (int step = 0; step < len; step++) {
      // next smallest gidx strictly greater than prev_g
      int si = -1;
      int32_t sg = 0;
      for (int i = 0; i < len; i++) {
        if (set_gidx[i] > prev_g && (si < 0 || set_gidx[i] < sg)) {
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

  // Embed gather: requant the winner's resident row. embed_sc cancels, same
  // math as llama_embed_select.cc.
  const int8_t *row = set_row + winner * kD;
  int32_t amax = 0;
  for (int i = 0; i < kD; i++) {
    int32_t a = row[i];
    if (a < 0)
      a = -a;
    if (a > amax)
      amax = a;
  }
  if (amax < 1)
    amax = 1;
  float inv = sw_recip((float)amax) * 127.0f; // 127/amax
  for (int i = 0; i < kD; i++)
    seed[i] = round_to_i8((float)row[i] * inv);
  float scale = (float)amax * set_scale[winner] * (1.0f / 127.0f);
  memcpy(seed + kD, &scale, 4);
  int32_t zero = 0;
  memcpy(seed + kD + 4, &zero, 4);
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
  memcpy(out + kD + 4, &token, 4);
}

} // extern "C"

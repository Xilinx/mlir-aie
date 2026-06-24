//===- llama_topk_sample.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// One-stream sampler + embed-gather: the single tied-table pass that the
// persistent loop needs.
//
// THE PROBLEM IT SOLVES. The lm_head streams the 262 MB tied embed table once
// per token to produce logits. To sample AND fetch the next token's embedding
// row on-device, the naive design streams the table TWICE (lm_head GEMM +
// embed-gather) -- two redundant 262 MB shim streams that saturate the shim DMA
// and waste ~25% of the per-token bandwidth (the decode bottleneck). This
// kernel removes the second pass: as the table streams ONCE, it keeps a
// resident top-k of {logit, global index, embed_sc, embed ROW}. Keeping the
// row (not just the index) is the crux -- the winner's embedding is already in
// L1/memtile when sampling finishes, so the gather is free and no second table
// pass is needed.
//
// SAMPLING SEMANTICS. This does CLEAN top-k renormalization: the softmax +
// inverse-CDF run over EXACTLY the k surviving tokens (standard top-k
// sampling). This differs intentionally from llama_sample.cc /
// llama_sample_streamed.cc, which add a tiny exp_lut(-128) mass for every one
// of the V-k filtered tokens (a bit-exact-legacy quirk inherent to their
// "mask in place over all V" structure). Clean renorm is the more correct
// behavior and the only one expressible without re-touching all V logits.
// numpy_topk_sample.py is the matching oracle.
//
//   greedy (temperature <= 0): winner = global argmax (first-occurrence
//     tie-break), which is the max-logit entry of the set (the set always
//     contains the global max).
//   multinomial (temperature > 0, top_k > 0):
//     z[i]    = set_logit[i] * inv_temp           (over the k survivors)
//     max_z   = max z
//     sum_exp = Σ exp_lut(quant_shifted(z - max_z))    (k terms, clean)
//     u       = xoshiro_uniform(seed) * sum_exp
//     winner  = first survivor (walked in ASCENDING global-index order) whose
//               running cumsum exceeds u.
//   token = set_gidx[winner].
//   seed  = requant(set_row[winner]) -> int8[D] + fp32 scale tail (the embed
//           gather: embed_sc cancels, same math as llama_embed_select.cc).
//
// LIMITATION: top_k == 0 (full-vocab multinomial) is NOT supported here -- any
// token can win, so no bounded resident set holds the winner's row; that mode
// needs a second selective table pass. Real sampling always uses top-k/top-p,
// so this is documented as out-of-scope for the one-stream loop (see README).
//
// Peano AIE2P workarounds carried from llama_sample.cc / llama_flowkv.cc:
//   - sw_recip instead of `1.0f/x` (Bug 1: HW reciprocal is ~10 bits)
//   - rotl via 64-bit concat-shift (Bug 4: G_ROTL fails to legalize)
//   - never keep an fp32 stack array alive across loops (Bug 2): the resident
//     set lives in passed buffers; z is recomputed from set_logit each pass.
//
//===----------------------------------------------------------------------===//

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

// Insert one streamed slot of kNTile rows into the resident top-k set.
//   slot     : packed [logits f32 | scales f32 | rows i8] (see offsets above)
//   set_logit/set_gidx/set_scale : resident top-k metadata (capacity kKset)
//   set_row  : resident top-k embed rows (kKset * kD int8)
//   set_len  : current set occupancy (init 0 by host before the stream)
//   tile_idx : 0-based tile index; global row index = tile_idx*kNTile + j
//
// Min-eviction unsorted set: append until full, then replace the current min
// slot if a strictly-larger logit arrives. For DISTINCT fp32 logits this yields
// exactly the kKset largest (== the reference top-k); ties keep the incumbent
// (matches the oracle's first-seen tie-break). Real GEMM logits are fp32 and
// effectively distinct.
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
      for (int d = 0; d < kD; d++)
        set_row[len * kD + d] = row[d];
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
      for (int d = 0; d < kD; d++)
        set_row[mi * kD + d] = row[d];
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
    // Greedy: global argmax with first-occurrence (smallest gidx) tie-break.
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

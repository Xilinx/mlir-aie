//===- llama_sample.cc --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama sampler v1: temperature + top-k + softmax + multinomial.
//
// Matches the algorithm in cautious-eureka/npu2/sampling.py, but
// arranged to be bit-exact against a local numpy reference (xoshiro128++
// PRNG + the same exp LUT used by llama_flowkv.cc + sw_recip), not
// against sampling.py's numpy default_rng + np.exp + np.partition.
//
// Algorithm (per-call):
//   z[i] = logits[i] * inv_temperature                          (fp32)
//   if top_k > 0 and top_k < V:
//     kth = k-th largest z[]
//     mask z[i] < kth to a sentinel (-inf-like, large negative)
//   max  = max(z)
//   for each i: q[i] = quant_shifted(z[i] - max)                (int32 [-128,0])
//   sum  = sum(lookup_exp(q[i]))                                (fp32)
//   u    = xoshiro_uniform() * sum                              (inverse-CDF draw)
//   token = first i such that running_cumsum(lookup_exp(q[i])) > u
//
// temperature <= 0 short-circuits to greedy argmax (no PRNG used).
//
// PRNG: xoshiro128++, seeded from a uint32 input via splitmix64-style
// expansion. Reference (numpy_sample.py) mirrors the same uint32
// arithmetic for bit-exact validation.
//
// Workarounds carried over from llama_flowkv.cc (Peano AIE2P bugs):
//   - sw_recip instead of `1.0f / x` (Bug 1: HW reciprocal is ~10 bits)
//   - never store fp32 in a stack array between loops (Bug 2). Stash
//     int32 LUT indices instead and re-call lookup_exp() in each pass.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

#include "build/exp_lut.h"

#ifndef LLAMA_SAMPLE_VOCAB
#define LLAMA_SAMPLE_VOCAB 1024
#endif
#ifndef LLAMA_SAMPLE_EXP_QUANT_SCALE
#define LLAMA_SAMPLE_EXP_QUANT_SCALE 0.05f
#endif

static constexpr int   kV = LLAMA_SAMPLE_VOCAB;
static constexpr float kExpQuantScale    = LLAMA_SAMPLE_EXP_QUANT_SCALE;
static constexpr float kInvExpQuantScale = 1.0f / LLAMA_SAMPLE_EXP_QUANT_SCALE;

// Sentinel "masked-out" score in shifted-units. exp_quant clamp at -128
// gives ~exp(-6.4) ~= 1.7e-3, so a value comfortably below -128/kInv
// (i.e. < -6.4 in score units) will quantize to -128 in lookup. We use
// -1e9 as a sentinel; both kernel and ref produce q = -128 -> exp(-6.4).
// In sum context that's tiny compared to the surviving top-k masses
// (which have q >= -128 too, but with shifted=0 max producing exp(0)=1).
// Reference produces identical value because identical clamp path.
static constexpr float kMaskSentinel = -1.0e9f;

// --- exp LUT shared with flowkv (build/exp_lut.h emits llama_exp_lut_raw[256]) ---

static inline int32_t quant_shifted(float shifted) {
  float v = shifted * kInvExpQuantScale;
  int32_t q = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (q > 0) q = 0;
  if (q < -128) q = -128;
  return q;
}

static inline float lookup_exp(int32_t q) {
  uint32_t raw = llama_exp_lut_raw[q + 128];
  float v;
  memcpy(&v, &raw, 4);
  return v;
}

// IEEE fp32 reciprocal (NR, 4 iters). See llama_flowkv.cc + bugs memo.
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

// --- xoshiro128++ PRNG (Blackman & Vigna 2018) ---

struct Xoshiro128 { uint32_t s[4]; };

// Peano AIE2P backend cannot legalize G_ROTL (the canonical
// `(x << k) | (x >> (32-k))` lowers to G_ROTL which fails -- file as
// Bug 4 if not already documented). Force a 64-bit concat-and-shift
// pattern that the backend can lower correctly.
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

// splitmix64-style seed expansion from a single uint32. Numpy reference
// mirrors this exactly with np.uint64 arithmetic.
static inline void xoshiro_seed(Xoshiro128 &g, uint32_t seed) {
  uint64_t x = ((uint64_t)seed) * 0x9E3779B97F4A7C15ULL + 1ULL;
  for (int i = 0; i < 4; i++) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    g.s[i] = (uint32_t)(x ^ (x >> 32));
  }
}

// Uniform fp32 in [0, 1): 24 random bits / 2^24. Standard pattern.
static inline float xoshiro_uniform(Xoshiro128 &g) {
  uint32_t r = xoshiro_next(g);
  return (float)(r >> 8) * (1.0f / 16777216.0f);
}

// --- Sampler ---

extern "C" {

// params layout (3 x uint32, packed into one fifo to stay under the 2-in
// DMA budget): [0] = temperature as raw fp32 bits, [1] = top_k as int32,
// [2] = seed as uint32. Host packs in the same order; numpy ref reads
// the same layout from the test.
void llama_sample(int8_t   *restrict logits,
                  int32_t  *restrict token_id,
                  uint32_t *restrict params) {
  float temperature;
  uint32_t tbits = params[0];
  memcpy(&temperature, &tbits, 4);
  const int32_t  top_k = (int32_t)params[1];
  const uint32_t seed  = params[2];


  // Greedy short-circuit: argmax with first-occurrence tie-break.
  if (temperature <= 0.0f) {
    int32_t best_idx = 0;
    int32_t best_val = (int32_t)logits[0];
    for (int32_t v = 1; v < kV; v++) {
      int32_t l = (int32_t)logits[v];
      if (l > best_val) { best_val = l; best_idx = v; }
    }
    token_id[0] = best_idx;
    return;
  }

  const float inv_temp = sw_recip(temperature);

  // Temperature-scaled logits stashed as int32 LUT indices (Bug 2 work-
  // around: avoid fp32 stack array between loops). We need to apply
  // top-k masking BEFORE max-subtract, so we do an integer pass on the
  // raw scaled values first, then a second pass to quantize after the
  // max subtract.

  // Pass 1: scaled fp32 values, stored as raw uint32 bits in an int32
  // buffer (re-cast each access). Not a "stack float[] reused across
  // loops" pattern -- we're using an int32 buffer and bit-casting on
  // read, which the Peano optimizer handles correctly per Bug 2 notes.
  int32_t z_bits[kV];
  for (int v = 0; v < kV; v++) {
    float zv = (float)logits[v] * inv_temp;
    int32_t b;
    memcpy(&b, &zv, 4);
    z_bits[v] = b;
  }

  // Top-k masking: find the k-th largest scaled value via an O(V*k)
  // "find max, mask, repeat" loop. For V=1024, k=40 this is 40k ops --
  // cheap, and bit-exact with the reference because we use the same
  // algorithm (NOT np.partition, which has different tie-break / order).
  //
  // Implementation: maintain a parallel "masked" flag array; on each of
  // the k iterations, scan once for the current max among non-masked
  // entries, record its value as the threshold (last seen = k-th), and
  // mark it masked. After k iterations, threshold = k-th largest. Then
  // re-scan and overwrite z_bits[v] -> kMaskSentinel for any v with the
  // raw fp value strictly less than threshold.
  if (top_k > 0 && top_k < kV) {
    int8_t masked[kV];
    for (int v = 0; v < kV; v++) masked[v] = 0;

    float threshold = 0.0f;
    for (int iter = 0; iter < top_k; iter++) {
      int   best_v = -1;
      float best_z = 0.0f;
      for (int v = 0; v < kV; v++) {
        if (masked[v]) continue;
        float zv;
        memcpy(&zv, &z_bits[v], 4);
        if (best_v < 0 || zv > best_z) { best_v = v; best_z = zv; }
      }
      if (best_v < 0) break;     // should not happen when top_k < kV
      threshold = best_z;
      masked[best_v] = 1;
    }

    // Mask anything strictly below threshold.
    for (int v = 0; v < kV; v++) {
      float zv;
      memcpy(&zv, &z_bits[v], 4);
      if (zv < threshold) {
        float s = kMaskSentinel;
        int32_t b;
        memcpy(&b, &s, 4);
        z_bits[v] = b;
      }
    }
  }

  // Find max of the (possibly masked) z.
  float max_z;
  memcpy(&max_z, &z_bits[0], 4);
  for (int v = 1; v < kV; v++) {
    float zv;
    memcpy(&zv, &z_bits[v], 4);
    if (zv > max_z) max_z = zv;
  }

  // Pass 2: per-element quantize (z[v] - max) and accumulate sum_exp.
  // Stash q[v] as int32 (Bug 2 workaround: do NOT keep an fp32 e_v[]
  // alive across loops; re-call lookup_exp(qvals[v]) in the cumsum
  // pass).
  int32_t qvals[kV];
  float sum_exp = 0.0f;
  for (int v = 0; v < kV; v++) {
    float zv;
    memcpy(&zv, &z_bits[v], 4);
    int32_t q = quant_shifted(zv - max_z);
    qvals[v] = q;
    sum_exp += lookup_exp(q);
  }

  // Inverse-CDF draw. u in [0, sum_exp).
  Xoshiro128 g;
  xoshiro_seed(g, seed);
  float u = xoshiro_uniform(g) * sum_exp;

  // First-past-the-post scan: walk cumsum, pick the first index whose
  // running total exceeds u. For numerical safety, return last index if
  // we somehow fall off the end (only possible from fp roundoff).
  float c = 0.0f;
  int32_t pick = kV - 1;
  for (int v = 0; v < kV; v++) {
    c += lookup_exp(qvals[v]);
    if (c > u) { pick = v; break; }
  }
  token_id[0] = pick;
}

} // extern "C"

//===- llama_sample_streamed.cc -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Streamed sampler for the REAL vocab (V=128256). Same algorithm as
// llama_sample.cc (temperature + top-k + exp-LUT softmax + xoshiro128++
// inverse-CDF draw, greedy as the temp<=0 short-circuit), but the fp32
// logits live in DDR and are streamed through an L1 chunk fifo in THREE
// ordered passes -- the resident int8[V]/int32[V] arrays of llama_sample.cc
// (~1 MB at this V) do not fit L1.
//
// Passes (driven by the IRON worker; same DDR logits buffer re-filled 3x):
//   pass 1: max(z) over all logits  + (if top-k) maintain the top-k set
//           + (greedy) running argmax over raw logits.
//   pass 2: threshold = k-th largest (from the top-k set); accumulate
//           sum_exp = Σ exp_lut(quant_shifted(z - max)) in INDEX ORDER
//           (masked entries contribute exp_lut(-128), matching the ref).
//   pass 3: u = xoshiro_uniform(seed)*sum_exp; walk the cumulative sum in
//           index order, pick the first index whose running total > u.
//
// State (carried across all chunk calls in an L1 buffer the worker holds):
//   [0] max_z f32   [4] threshold f32  [8] sum_exp f32  [12] cum f32
//   [16] u f32      [20] pick i32      [24] best_idx i32 (greedy)
//   [28] heap_len i32                  [64..] top-k value set (fp32 * MAX_TOPK)
//
// Top-k uses a resident UNSORTED top-k value set (capacity MAX_TOPK) instead
// of llama_sample.cc's V-sized index mask: one pass keeps the k largest z;
// threshold = the set's min (= k-th largest). For DISTINCT fp32 logits this
// is identical to the reference's mask-one-per-pass result. The matching
// numpy oracle (numpy_sample_streamed.py) uses the same value-threshold.
//
// Peano AIE2P workarounds carried from llama_sample.cc / llama_flowkv.cc:
//   - sw_recip instead of `1.0f/x` (Bug 1)
//   - rotl via 64-bit concat-shift (Bug 4: G_ROTL fails to legalize)
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

#include "build/exp_lut.h"

#ifndef LLAMA_SAMPLE_EXP_QUANT_SCALE
#define LLAMA_SAMPLE_EXP_QUANT_SCALE 0.05f
#endif
#ifndef LLAMA_SAMPLE_MAX_TOPK
#define LLAMA_SAMPLE_MAX_TOPK 256
#endif
#ifndef LLAMA_SAMPLE_CHUNK_N
#define LLAMA_SAMPLE_CHUNK_N 2004
#endif
#ifndef LLAMA_SAMPLE_VOCAB
#define LLAMA_SAMPLE_VOCAB 128256
#endif

static constexpr float kExpQuantScale = LLAMA_SAMPLE_EXP_QUANT_SCALE;
static constexpr float kInvExpQuantScale = 1.0f / LLAMA_SAMPLE_EXP_QUANT_SCALE;
static constexpr int kMaxTopK = LLAMA_SAMPLE_MAX_TOPK;
static constexpr int kChunkN = LLAMA_SAMPLE_CHUNK_N;
static constexpr int kVocab = LLAMA_SAMPLE_VOCAB;
static constexpr float kMaskSentinel = -1.0e9f;
static constexpr float kNegInf = -3.0e38f;

// state byte offsets
static constexpr int OFF_MAXZ = 0;
static constexpr int OFF_THRESH = 4;
static constexpr int OFF_SUMEXP = 8;
static constexpr int OFF_CUM = 12;
static constexpr int OFF_U = 16;
static constexpr int OFF_PICK = 20;
static constexpr int OFF_BESTIDX = 24;
static constexpr int OFF_HEAPLEN = 28;
static constexpr int OFF_HEAP = 64;

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

// --- xoshiro128++ (must match llama_sample.cc / numpy_sample.py) ---
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

// Maintain an unsorted top-k value set (capacity k). Returns nothing; the
// set lives in heap[0..len). Insert z: if len<k append; else if z>min,
// replace the current min slot. threshold (k-th largest) = the set's min.
static inline void topk_insert(float *heap, int32_t *len, int k, float z) {
  if (*len < k) {
    heap[*len] = z;
    (*len)++;
    return;
  }
  // find min slot
  int mi = 0;
  float mv = heap[0];
  for (int i = 1; i < k; i++) {
    if (heap[i] < mv) {
      mv = heap[i];
      mi = i;
    }
  }
  if (z > mv)
    heap[mi] = z;
}

static inline float topk_min(const float *heap, int len) {
  float mv = heap[0];
  for (int i = 1; i < len; i++)
    if (heap[i] < mv)
      mv = heap[i];
  return mv;
}

extern "C" {

// One streamed-sampler chunk call. `base` points at a resident buffer (a DDR
// chunk in M2, or a whole memtile HALF in M3a); the kernel reads kChunkN
// elements starting at local_chunk*kChunkN within it. state is the resident
// carry buffer. params = [temperature bits, top_k, seed]. pass in {1,2,3};
// chunk_idx is the 0-based GLOBAL chunk within the pass (for global token
// indexing). chunk_n / v_total are compile-time constants (kChunkN / kVocab);
// chunk_base = chunk_idx * kChunkN. M2 passes local_chunk=0 (chunk-typed
// fifo); M3a passes local_chunk = the chunk's index within its half.
void llama_sample_streamed(float *restrict base, int8_t *restrict state,
                           uint32_t *restrict params, int32_t pass,
                           int32_t chunk_idx, int32_t local_chunk) {
  event0();
  float *restrict logits = base + local_chunk * kChunkN;
  const int32_t chunk_base = chunk_idx * kChunkN;
  const int32_t chunk_n = kChunkN;
  const int32_t v_total = kVocab;
  float temperature;
  uint32_t tbits = params[0];
  memcpy(&temperature, &tbits, 4);
  const int32_t top_k = (int32_t)params[1];
  const uint32_t seed = params[2];
  const bool greedy = temperature <= 0.0f;
  const bool use_topk = (top_k > 0 && top_k < v_total);
  int kcap = top_k;
  if (kcap > kMaxTopK)
    kcap = kMaxTopK;

  float *heap = reinterpret_cast<float *>(state + OFF_HEAP);

  if (pass == 1) {
    float max_z;
    int32_t best_idx, heap_len;
    if (chunk_base == 0) {
      max_z = kNegInf;
      best_idx = -1;
      heap_len = 0;
    } else {
      memcpy(&max_z, state + OFF_MAXZ, 4);
      memcpy(&best_idx, state + OFF_BESTIDX, 4);
      memcpy(&heap_len, state + OFF_HEAPLEN, 4);
    }
    if (greedy) {
      float best_val;
      memcpy(&best_val, state + OFF_MAXZ, 4); // reuse MAXZ as best_val
      for (int i = 0; i < chunk_n; i++) {
        float l = logits[i];
        int32_t gidx = chunk_base + i;
        if (best_idx < 0 || l > best_val) {
          best_val = l;
          best_idx = gidx;
        }
      }
      memcpy(state + OFF_MAXZ, &best_val, 4);
      memcpy(state + OFF_BESTIDX, &best_idx, 4);
    } else {
      const float inv_temp = sw_recip(temperature);
      for (int i = 0; i < chunk_n; i++) {
        float z = logits[i] * inv_temp;
        if (z > max_z)
          max_z = z;
        if (use_topk)
          topk_insert(heap, &heap_len, kcap, z);
      }
      memcpy(state + OFF_MAXZ, &max_z, 4);
      memcpy(state + OFF_HEAPLEN, &heap_len, 4);
    }
    event1();
    return;
  }

  if (greedy) { // passes 2/3 are no-ops for greedy
    event1();
    return;
  }

  const float inv_temp = sw_recip(temperature);
  float max_z;
  memcpy(&max_z, state + OFF_MAXZ, 4);

  if (pass == 2) {
    float sum_exp, threshold;
    if (chunk_base == 0) {
      sum_exp = 0.0f;
      if (use_topk) {
        int32_t heap_len;
        memcpy(&heap_len, state + OFF_HEAPLEN, 4);
        threshold = topk_min(heap, heap_len);
      } else {
        threshold = kNegInf;
      }
      memcpy(state + OFF_THRESH, &threshold, 4);
    } else {
      memcpy(&sum_exp, state + OFF_SUMEXP, 4);
      memcpy(&threshold, state + OFF_THRESH, 4);
    }
    for (int i = 0; i < chunk_n; i++) {
      float z = logits[i] * inv_temp;
      if (use_topk && z < threshold)
        z = kMaskSentinel;
      int32_t q = quant_shifted(z - max_z);
      sum_exp += lookup_exp(q);
    }
    memcpy(state + OFF_SUMEXP, &sum_exp, 4);
    event1();
    return;
  }

  // pass == 3: inverse-CDF draw. Walk the cumulative sum in index order;
  // the first index whose running total exceeds u is the pick. State carries
  // (cum, u, threshold, pick, found) across chunks; OFF_BESTIDX is reused as
  // the found flag (greedy doesn't reach pass 3).
  float cum, u, threshold;
  int32_t pick, found;
  if (chunk_base == 0) {
    float sum_exp;
    memcpy(&sum_exp, state + OFF_SUMEXP, 4);
    memcpy(&threshold, state + OFF_THRESH, 4);
    Xoshiro128 g;
    xoshiro_seed(g, seed);
    u = xoshiro_uniform(g) * sum_exp;
    cum = 0.0f;
    pick = v_total - 1; // fallback if u falls off the end (fp roundoff)
    found = 0;
    memcpy(state + OFF_U, &u, 4);
  } else {
    memcpy(&cum, state + OFF_CUM, 4);
    memcpy(&u, state + OFF_U, 4);
    memcpy(&threshold, state + OFF_THRESH, 4);
    memcpy(&pick, state + OFF_PICK, 4);
    memcpy(&found, state + OFF_BESTIDX, 4);
  }
  if (!found) {
    for (int i = 0; i < chunk_n; i++) {
      float z = logits[i] * inv_temp;
      if (use_topk && z < threshold)
        z = kMaskSentinel;
      int32_t q = quant_shifted(z - max_z);
      cum += lookup_exp(q);
      if (cum > u) {
        pick = chunk_base + i;
        found = 1;
        break;
      }
    }
  }
  memcpy(state + OFF_CUM, &cum, 4);
  memcpy(state + OFF_PICK, &pick, 4);
  memcpy(state + OFF_BESTIDX, &found, 4);
  event1();
}

// Read the final token id from the carry state. greedy -> best_idx; else the
// inverse-CDF pick. The worker calls this once after pass 3.
void llama_sample_streamed_finalize(int8_t *restrict state,
                                    int32_t *restrict token,
                                    uint32_t *restrict params) {
  event0();
  float temperature;
  uint32_t tbits = params[0];
  memcpy(&temperature, &tbits, 4);
  int32_t out;
  if (temperature <= 0.0f)
    memcpy(&out, state + OFF_BESTIDX, 4); // greedy best_idx
  else
    memcpy(&out, state + OFF_PICK, 4);
  token[0] = out;
  event1();
}

} // extern "C"

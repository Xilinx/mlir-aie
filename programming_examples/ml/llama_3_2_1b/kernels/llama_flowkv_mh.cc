//===- llama_flowkv_mh.cc -----------------------------------*- C++ -*-===//
// Phase 7a: multi-head GQA flowkv -- REP Q heads sharing 1 KV head,
// fused qk+sv per head, all in one kernel call.
//
// Arithmetic mirrors llama_flowkv.cc qk + sv exactly so the per-head
// computation is bit-identical to running the existing single-head
// kernels REP times. Reuses quant_shifted / lookup_exp / sw_recip /
// round_to_i8 (copied verbatim from llama_flowkv.cc).
//
// Buffer layout (all int8 unless noted):
//   q_chunk:   REP*kHD bytes of q data, then a REP*8-byte tail with
//              per-head [q_scale_h: fp32, sv_inv_out_scale_h: fp32].
//   k_one:     4 bytes k_scale (fp32) + kT*kHD bytes of k cache.
//   v_one:     4 bytes v_scale (fp32) + kT*kHD bytes of v cache.
//   out_chunk: REP*kHD bytes of int8 sv outputs (REP heads concatenated).
//
// One worker holds one KV head's cache (kT*kHD = 8192 bytes each for
// k and v at T=128, HD=64) plus light scratch -- comfortably under the
// 64 KB AIE2P L1 budget.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "build/exp_lut.h"

#ifndef LLAMA_FLOWKV_HEAD_DIM
#define LLAMA_FLOWKV_HEAD_DIM 64
#endif
#ifndef LLAMA_FLOWKV_T
#define LLAMA_FLOWKV_T 128
#endif
#ifndef LLAMA_FLOWKV_REP
#define LLAMA_FLOWKV_REP 4
#endif
#ifndef LLAMA_FLOWKV_EXP_QUANT_SCALE
#define LLAMA_FLOWKV_EXP_QUANT_SCALE 0.05f
#endif

static constexpr int kHD = LLAMA_FLOWKV_HEAD_DIM;
static constexpr int kT = LLAMA_FLOWKV_T;
static constexpr int kREP = LLAMA_FLOWKV_REP;
static constexpr float kExpQuantScale = LLAMA_FLOWKV_EXP_QUANT_SCALE;
static constexpr float kInvExpQuantScale = 1.0f / LLAMA_FLOWKV_EXP_QUANT_SCALE;

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX)
    r = I8_MAX;
  if (r < I8_MIN)
    r = I8_MIN;
  return (int8_t)r;
}

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

// Newton-Raphson software reciprocal -- Peano AIE2P `/` is HW
// reciprocal approximation, not IEEE-correct (Bug 1).
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

// Forward decl of the inner kernel with explicit T_used (causal-mask
// support). The combined wrapper reads T_used + per-slot scale arrays from
// kv_one's prefix and forwards. k_slot/v_slot are per-cache-slot fp32 scale
// arrays (one entry per cached position) -- each cached K/V position is
// dequantized by ITS OWN scale (fixes the per-head-scalar bug where the whole
// cache was dequantized with the latest token's scale).
extern "C" void llama_flowkv_mh(int8_t *restrict q_chunk,
                                float *restrict k_slot, int8_t *restrict k_body,
                                float *restrict v_slot, int8_t *restrict v_body,
                                int8_t *restrict out_chunk, int T_used);

// Per-slot kv_one layout (per KV head):
//   [0..4]:                          T_used (int32)
//   [4..8]:                          pad (so the scale region starts at 8)
//   [8 .. 8+kT*4]:                   k_slot_scales (kT fp32)
//   [8+kT*4 .. 8+kT*4+kT*kHD]:       k body (kT * kHD int8)
//   [.. + kT*4]:                     v_slot_scales (kT fp32)
//   [.. + kT*kHD]:                   v body (kT * kHD int8)
// Total per head = 8 + 2*(kT*4 + kT*kHD).
constexpr int kTUsedPrefix = 8;
extern "C" void llama_flowkv_mh_kvc(int8_t *restrict q_chunk,
                                    int8_t *restrict kv_one,
                                    int8_t *restrict out_chunk) {
  int T_used;
  memcpy(&T_used, kv_one, 4);
  if (T_used <= 0)
    T_used = 1;
  if (T_used > kT)
    T_used = kT;
  constexpr int kSlotBytes = kT * 4;
  constexpr int kBodyBytes = kT * kHD;
  constexpr int kKHalf = kSlotBytes + kBodyBytes;
  int8_t *k_region = kv_one + kTUsedPrefix;
  int8_t *v_region = k_region + kKHalf;
  llama_flowkv_mh(q_chunk, reinterpret_cast<float *>(k_region),
                  k_region + kSlotBytes, reinterpret_cast<float *>(v_region),
                  v_region + kSlotBytes, out_chunk, T_used);
}

extern "C" void llama_flowkv_mh(int8_t *restrict q_chunk,
                                float *restrict k_slot, int8_t *restrict k_body,
                                float *restrict v_slot, int8_t *restrict v_body,
                                int8_t *restrict out_chunk, int T_used) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  static_assert(kHD == 64, "qk_scale hardcoded for head_dim=64");
  constexpr float kInvSqrtHD = 0.125f;

  // Per-head scale tail layout: tail[h*8..h*8+4] = q_scale_h,
  //                             tail[h*8+4..h*8+8] = sv_inv_out_scale_h.
  int8_t *tail = q_chunk + kREP * kHD;

  // Scratch buffers reused across all REP iterations. NOT a float[kT]
  // (Bug 2: stack fp32 array between loops can be coalesced/corrupted
  // by the optimizer); int32 qvals is the same pattern the single-head
  // kernel uses successfully.
  float scores[kT];
  int32_t qvals[kT];

  for (int h = 0; h < kREP; h++) {
    float q_scale, sv_inv_out_scale;
    memcpy(&q_scale, tail + h * 8 + 0, 4);
    memcpy(&sv_inv_out_scale, tail + h * 8 + 4, 4);

    int8_t *restrict q_h = q_chunk + h * kHD;
    int8_t *restrict out_h = out_chunk + h * kHD;

    // 1) Scores: int32 dot -> per-slot combined float multiply. Each cached
    // position uses its OWN k_slot[i] scale (the per-slot KV fix). Causal
    // mask: only the first T_used positions. Scale order (q_scale*k_slot[i])
    // * kInvSqrtHD mirrors the numpy reference (and the per-head predecessor's
    // q_scale*k_scale*kInvSqrtHD). NOTE: with varying per-slot scales this is
    // NOT byte-exact vs numpy -- Peano fp32 mul differs by ~1 ULP on a few
    // positions, flipping an exp-LUT bucket (<=2 LSB on the output). Proven
    // quality-neutral; see numpy_attention_perslot_lut + test_layer_mh.
    float max_s = -1e30f;
    for (int i = 0; i < T_used; i++) {
      int32_t dot = 0;
      for (int d = 0; d < kHD; d++) {
        dot += (int32_t)q_h[d] * (int32_t)k_body[i * kHD + d];
      }
      float s = (float)dot * ((q_scale * k_slot[i]) * kInvSqrtHD);
      scores[i] = s;
      if (s > max_s)
        max_s = s;
    }

    // 2) Shift, quantize via LUT-step, sum.
    float sum = 0.0f;
    for (int i = 0; i < T_used; i++) {
      int32_t q = quant_shifted(scores[i] - max_s);
      qvals[i] = q;
      sum += lookup_exp(q);
    }
    const float inv_sum = sw_recip(sum);

    // 3) sv: probs @ V then requant. Per-slot V scale folds into the inner
    // term: acc += (p * v_slot[i]) * v[i,j]. j-outer / i-inner matches the
    // single-head kernel; probs[i] recomputed from qvals[i] each i to avoid
    // Bug 2 (fp32 stack array between loops).
    for (int j = 0; j < kHD; j++) {
      float acc = 0.0f;
      for (int i = 0; i < T_used; i++) {
        float p = lookup_exp(qvals[i]) * inv_sum;
        acc += p * v_slot[i] * (float)v_body[i * kHD + j];
      }
      out_h[j] = round_to_i8(acc * sv_inv_out_scale);
    }
  }

  event1();
}

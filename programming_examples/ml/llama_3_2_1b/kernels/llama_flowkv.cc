//===- llama_flowkv.cc --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama attention pair: non-chunked full-softmax split across two CTs,
// bit-exact against numpy -- int32 dot products scaled by one combined
// qk_scale, exp via a 256-entry LUT (gen_exp_lut.py, not aie::exp2,
// which approximates differently), exact 1.0f/sum (not aie::inv, a HW
// approximation), fp32 probs across the CT0->CT1 stream (not bf16).
//
// CT0 (llama_flowkv_qk):  Q_i8 + K_packed_i8 -> probs_fp32 (size T)
// CT1 (llama_flowkv_sv):  V_packed_i8 + probs_fp32 -> out_i8 (head_dim)
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "build/exp_lut.h"

#ifndef LLAMA_FLOWKV_HEAD_DIM
#define LLAMA_FLOWKV_HEAD_DIM 64
#endif
#ifndef LLAMA_FLOWKV_T
#define LLAMA_FLOWKV_T 16
#endif
#ifndef LLAMA_FLOWKV_EXP_QUANT_SCALE
// LUT step in score-units. lut[k] = exp((k - 128) * EXP_QUANT_SCALE).
// 0.05 covers shifted scores down to -128*0.05 = -6.4
// (exp(-6.4) = 1.7e-3, safely below the dominant terms).
#define LLAMA_FLOWKV_EXP_QUANT_SCALE 0.05f
#endif

static constexpr int kHD = LLAMA_FLOWKV_HEAD_DIM;
static constexpr int kT = LLAMA_FLOWKV_T;
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

// Quantize a shifted (<=0) score to int8 with the LUT step, clamp to
// [-128, 0]; matches the test's numpy reference.
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

// Software fp32 reciprocal: Peano lowers `1.0f / a` on AIE2P to a HW
// reciprocal approximation (~10 bits accurate, ~0.5% relative error),
// NOT IEEE-correct fp32 division. Use Newton-Raphson over a bit-hack
// initial guess to converge to IEEE fp32 precision (matches numpy's
// 1.0 / a exactly when numpy uses the same NR steps).
//
// x_{n+1} = x * (2 - a * x)  -- quadratic convergence on 1/a.
// After 4 NR iterations from a bit-hack guess (~6 bit initial), the
// result is bit-identical to IEEE fp32(1/a) for our input range.
static inline float sw_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits; // magic constant for 1/x estimate
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

extern "C" {

// CT0: compute fp32 softmax probabilities for one query against the K cache.
void llama_flowkv_qk(int8_t *restrict q_i8, int8_t *restrict k_i8,
                     float *restrict probs_out, float q_scale, float k_scale) {
  event0();

  // AIE2P fp default rounding is NOT round-to-nearest-even -- explicitly
  // set conv_even (banker's rounding, matches numpy's astype semantics).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // 1/sqrt(64) = 0.125 (Llama 3.2 1B head_dim is always 64). For other
  // head_dim, precompute in the Python build and pass as a scalar arg.
  static_assert(kHD == 64, "qk_scale hardcoded for head_dim=64");
  constexpr float kInvSqrtHD = 0.125f;
  const float qk_scale = q_scale * k_scale * kInvSqrtHD;

  // 1) Scores: int32 dot, then ONE combined float multiply.
  float scores[kT];
  float max_s = -1e30f;
  for (int i = 0; i < kT; i++) {
    int32_t dot = 0;
    for (int d = 0; d < kHD; d++) {
      dot += (int32_t)q_i8[d] * (int32_t)k_i8[i * kHD + d];
    }
    float s = (float)dot * qk_scale;
    scores[i] = s;
    if (s > max_s)
      max_s = s;
  }

  // 2) Shift + LUT-exp + sum. Store quantized q in qvals so we don't
  // need to re-quantize in the probs loop. Don't store exp_v -- on
  // AIE2P, indexing into a stack float[T] array between loops can get
  // optimized into vec ops that re-load corrupted/coalesced values.
  // We re-lookup exp_v from qvals in the probs loop instead.
  int32_t qvals[kT];
  float sum = 0.0f;
  for (int i = 0; i < kT; i++) {
    int32_t q = quant_shifted(scores[i] - max_s);
    qvals[i] = q;
    sum += lookup_exp(q);
  }

  // 3) Normalize and emit fp32 probs (sw_recip, not `1.0f / sum` -- see
  // its definition above).
  const float inv_sum = sw_recip(sum);
  for (int i = 0; i < kT; i++) {
    probs_out[i] = lookup_exp(qvals[i]) * inv_sum;
  }

  event1();
}

// CT1: probs @ V_dequant, requant to int8.
void llama_flowkv_sv(int8_t *restrict v_i8, float *restrict probs_in,
                     int8_t *restrict out_i8, float v_scale,
                     float inv_out_scale) {
  event0();

  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int j = 0; j < kHD; j++) {
    float acc = 0.0f;
    for (int i = 0; i < kT; i++) {
      acc += probs_in[i] * (float)v_i8[i * kHD + j];
    }
    // (inv_out_scale is HW-supplied as a precomputed reciprocal from
    // host, so this is a regular fp32 multiply -- no division here.)
    out_i8[j] = round_to_i8(acc * v_scale * inv_out_scale);
  }

  event1();
}

// Dynamic-scale variants (Phase 6c.5b.3).
//   _dyn qk: `q` is sized kHD+8 (trailing 8 B = q_scale + spare, written
//           by rope_int8_dyn passthrough). `k` is sized kT*kHD+8 (leading
//           8 B = k_scale + spare). Kernel reads scales from those tails
//           and bodies from the rest.
//   _dyn sv: `v` is sized kT*kHD+8 (leading 8 B = v_scale + inv_out_scale).
void llama_flowkv_qk_dyn(int8_t *restrict q_i8, int8_t *restrict k_i8,
                         float *restrict probs_out) {
  float q_scale, k_scale;
  memcpy(&q_scale, q_i8 + kHD, 4);
  memcpy(&k_scale, k_i8, 4);
  llama_flowkv_qk(q_i8, k_i8 + 8, probs_out, q_scale, k_scale);
}

void llama_flowkv_sv_dyn(int8_t *restrict v_i8, float *restrict probs_in,
                         int8_t *restrict out_i8) {
  float v_scale, inv_out_scale;
  memcpy(&v_scale, v_i8, 4);
  memcpy(&inv_out_scale, v_i8 + 4, 4);
  llama_flowkv_sv(v_i8 + 8, probs_in, out_i8, v_scale, inv_out_scale);
}

} // extern "C"

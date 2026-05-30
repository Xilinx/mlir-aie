//===- llama_flowkv.cc --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Llama attention pair v0: non-chunked full-softmax split across two CTs.
//
//   CT0 (llama_flowkv_qk):  Q_i8, K_packed -> probs_bf16
//     - dequant Q (head_dim,)  and K (t, head_dim) to bf16
//     - scores[i] = dot(Q, K[i]) * (1/sqrt(head_dim))
//     - softmax: subtract max, aie::exp2 via the log2e trick (matches
//       aie_kernels/aie2p/bf16_exp.cc), divide by sum (aie::inv)
//     - emit probs[i] in bf16 across the CT0 -> CT1 ObjectFifo
//
//   CT1 (llama_flowkv_sv):  V_packed, probs_bf16 -> out_i8
//     - dequant V (t, head_dim) to bf16
//     - out[j] = sum_i probs[i] * V[i, j]
//     - requant out to int8 with out_scale
//
// Chunked online-softmax (the "flowkv" name) is a v1 follow-up; this
// version replaces the cautious-eureka chunked design with the simpler
// "compute full softmax once" pattern (correct, slower per-step but
// avoids running-max correction state).
//
// K and V are packed payloads (entire KV cache for this head delivered
// in one ObjectFifo each).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LLAMA_FLOWKV_HEAD_DIM
#define LLAMA_FLOWKV_HEAD_DIM 64
#endif
#ifndef LLAMA_FLOWKV_T
#define LLAMA_FLOWKV_T 16
#endif

static constexpr int kHD = LLAMA_FLOWKV_HEAD_DIM;
static constexpr int kT  = LLAMA_FLOWKV_T;
static constexpr int kN  = 16;  // bf16 vector lane count

static constexpr float kLog2e = 1.44269504089f;
static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX) r = I8_MAX;
  if (r < I8_MIN) r = I8_MIN;
  return (int8_t)r;
}

extern "C" {

// CT0 of the attention pair: compute softmax probabilities for one
// query against the K cache.
void llama_flowkv_qk(int8_t *restrict q_i8, int8_t *restrict k_i8,
                     bfloat16 *restrict probs_out,
                     float q_scale, float k_scale) {
  event0();

  // 1/sqrt(64) = 0.125 (Llama 3.2 1B head_dim is always 64). For other
  // head_dim, precompute in the Python build and pass as a scalar arg.
  static_assert(kHD == 64, "qk_scale hardcoded for head_dim=64");
  constexpr float kInvSqrtHD = 0.125f;
  const float qk_scale = q_scale * k_scale * kInvSqrtHD;

  // Score per key: dot(Q, K[i]) * qk_scale.
  // Sum-of-products in int32; one scale at the end.
  float scores[kT];
  float max_s = -1e30f;
  for (int i = 0; i < kT; i++) {
    int32_t dot = 0;
    for (int d = 0; d < kHD; d++) {
      dot += (int32_t)q_i8[d] * (int32_t)k_i8[i * kHD + d];
    }
    float s = (float)dot * qk_scale;
    scores[i] = s;
    if (s > max_s) max_s = s;
  }

  // Subtract max, exp via 2^(x * log2e), sum.
  float sum = 0.0f;
  float exp_cache[kT];
  for (int i = 0; i < kT; i++) {
    float shifted = (scores[i] - max_s) * kLog2e;
    // Scalar exp2: __exp2f isn't always available on peano; use the
    // identity exp2(x) = e^(x * ln2) via... no, that's circular.
    // For first version, use ldexpf-based decomposition: exp2(x) =
    // ldexp(2^(x - floor(x)), floor(x)) where 2^frac is a small poly.
    // But that's heavier than worth it here -- the per-element exp can
    // be vectorized via aie::exp2<bfloat16> in a follow-up. For v0
    // just call the AIE2P scalar 2^x intrinsic via aie::exp2 on a
    // vector-of-1.
    aie::vector<float, 16> v;
    v[0] = shifted;
    auto e_v = aie::exp2<bfloat16>(v);
    float e = (float)e_v[0];
    exp_cache[i] = e;
    sum += e;
  }

  const float inv_sum = aie::inv(sum);  // HW reciprocal
  for (int i = 0; i < kT; i++) {
    probs_out[i] = (bfloat16)(exp_cache[i] * inv_sum);
  }

  event1();
}

// CT1 of the attention pair: multiply the softmax probs by V to get
// the attention output, then requant to int8.
void llama_flowkv_sv(int8_t *restrict v_i8, bfloat16 *restrict probs_in,
                     int8_t *restrict out_i8,
                     float v_scale, float inv_out_scale) {
  event0();

  // out[j] = sum_i probs[i] * V[i, j]_dequant
  //        = sum_i probs[i] * v_i8[i, j] * v_scale
  // For first version, scalar accumulate per output channel.
  for (int j = 0; j < kHD; j++) {
    float acc = 0.0f;
    for (int i = 0; i < kT; i++) {
      acc += (float)probs_in[i] * (float)v_i8[i * kHD + j];
    }
    out_i8[j] = round_to_i8(acc * v_scale * inv_out_scale);
  }

  event1();
}

} // extern "C"

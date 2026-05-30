//===- llama_rmsnorm_int8.cc -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Production-shaped RMSNorm: int8 in, int8 out, bf16-internal. Per
// cautious-eureka/docs/capacity_findings.md: between ops the dataflow
// is int8, but transcendentals (here aie::invsqrt) need float, and
// AIE2P DMA cannot type-convert -- so the int8 -> bf16 dequant and
// bf16 -> int8 SRS-style requant both live INSIDE the kernel.
//
//   x_f   = x_i8 * act_scale_in        (per-element dequant)
//   var   = mean(x_f^2)
//   inv_rms = aie::invsqrt(var + eps)   (HW intrinsic)
//   y_f   = x_f * inv_rms * gamma[i]
//   y_i8  = clamp(round(y_f * inv_act_scale_out), -128, 127)
//
// Equivalent (folding act_scale_in into the combined scalar):
//   combined = act_scale_in * inv_rms * inv_act_scale_out
//   y_i8     = clamp(round(x_i8 * combined * gamma[i]), -128, 127)
//
// The pass-1 sum-of-squares is computed entirely in float to keep the
// invsqrt approximation faithful. Pass-2 is vec bf16 multiply + scalar
// rounding cast (a vectorized round-and-narrow path is a follow-up).
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef LLAMA_RMSNORM_COLS
#define LLAMA_RMSNORM_COLS 2048
#endif
#ifndef LLAMA_RMSNORM_N
#define LLAMA_RMSNORM_N 16
#endif

static constexpr float kEps = 1e-5f;
static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  // banker's-rounding-equivalent for our range; matches Python round-half-
  // away-from-zero semantics commonly used in quant references. Keep it
  // simple and explicit; bit-exactness reference must use the same rule.
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX) r = I8_MAX;
  if (r < I8_MIN) r = I8_MIN;
  return (int8_t)r;
}

extern "C" {

void llama_rmsnorm_int8(int8_t *restrict x, bfloat16 *restrict gamma,
                        int8_t *restrict y,
                        float act_scale_in, float inv_act_scale_out) {
  event0();

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  constexpr int kN = LLAMA_RMSNORM_N;
  constexpr int kChunks = kCols / kN;

  // Pass 1: sum of squares in float. Vectorized over kN-lane bf16
  // chunks. Convert int8 input -> bf16, square, accumulate.
  ::aie::accum<acc32, kN> acc = ::aie::zeros<acc32, kN>();
  ::aie::vector<float, kN> add_res = ::aie::zeros<float, kN>();
  for (int i = 0; i < kChunks; i++) {
    // int8 -> bf16 unpack. AIE2P has aie::to_float / aie::to_bfloat16.
    ::aie::vector<int8, kN> xi = ::aie::load_v<kN>(x + i * kN);
    ::aie::vector<bfloat16, kN> xb = ::aie::to_float<bfloat16>(xi);
    ::aie::vector<float, kN> sq = ::aie::mul_square(xb);
    acc = ::aie::add(add_res, sq);
    add_res = acc.template to_vector<float>();
  }
  float sum_sq = ::aie::reduce_add(add_res);

  // Fold act_scale_in into the variance (sum_sq is over int8 values;
  // the real x_f^2 is act_scale_in^2 * x_i8^2).
  float var = (sum_sq * act_scale_in * act_scale_in) / (float)kCols;
  float inv_rms = aie::invsqrt(var + kEps);

  // Combined per-output multiplier:
  //   y_f = x_i8 * act_scale_in * inv_rms * gamma[i]
  //   y_i8 = round(y_f * inv_act_scale_out)
  float combined = act_scale_in * inv_rms * inv_act_scale_out;
  ::aie::vector<bfloat16, kN> combined_v =
      ::aie::broadcast<bfloat16, kN>(static_cast<bfloat16>(combined));

  // Pass 2: dequant + scale + gamma + requant. Vec mul for the bf16
  // path; scalar rounding cast for the final int8 store. The vec
  // round-and-narrow path is a follow-up optimization.
  for (int i = 0; i < kChunks; i++) {
    ::aie::vector<int8, kN> xi = ::aie::load_v<kN>(x + i * kN);
    ::aie::vector<bfloat16, kN> xb = ::aie::to_float<bfloat16>(xi);
    ::aie::vector<bfloat16, kN> gv = ::aie::load_v<kN>(gamma + i * kN);
    ::aie::vector<bfloat16, kN> scaled = ::aie::mul(xb, combined_v);
    ::aie::vector<bfloat16, kN> out_bf = ::aie::mul(scaled, gv);

    for (int j = 0; j < kN; j++) {
      y[i * kN + j] = round_to_i8(static_cast<float>(out_bf[j]));
    }
  }

  event1();
}

} // extern "C"

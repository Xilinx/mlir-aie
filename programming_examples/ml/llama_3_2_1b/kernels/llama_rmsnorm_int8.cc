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
#include <string.h>

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
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX) r = I8_MAX;
  if (r < I8_MIN) r = I8_MIN;
  return (int8_t)r;
}

// Pure IEEE fp32 invsqrt: Quake-III magic-constant initial guess + 2
// Newton-Raphson refinement iterations. Bit-identical to the numpy
// reference's sw_invsqrt -- no HW intrinsic approximation gap.
static inline float sw_invsqrt(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x5f3759df - (bits >> 1);
  float x;
  memcpy(&x, &bits, 4);
  x = x * (1.5f - 0.5f * a * x * x);
  x = x * (1.5f - 0.5f * a * x * x);
  return x;
}

extern "C" {

void llama_rmsnorm_int8(int8_t *restrict x, bfloat16 *restrict gamma,
                        int8_t *restrict y,
                        float act_scale_in, float inv_act_scale_out) {
  event0();

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  constexpr int kN = LLAMA_RMSNORM_N;
  constexpr int kChunks = kCols / kN;

  // Pass 1: exact int32 sum-of-squares over int8 inputs. int8 * int8
  // = int16; 2048 lanes worst-case = 2048 * 16384 = 33M, fits in int32.
  // Using exact int sums avoids any fp accumulation-order rounding so
  // the numpy reference can compute the identical value.
  int32_t sum_sq_i32 = 0;
  for (int i = 0; i < kCols; i++) {
    int32_t xi = x[i];
    sum_sq_i32 += xi * xi;
  }

  // Fold act_scale_in into the variance: x_f^2 = act_scale_in^2 * x_i8^2.
  float var = (float)sum_sq_i32 * act_scale_in * act_scale_in / (float)kCols;
  float inv_rms = sw_invsqrt(var + kEps);

  // Combined per-output multiplier (single fp32 scalar):
  //   y_f = x_i8 * act_scale_in * inv_rms * gamma[i]
  //   y_i8 = round(y_f * inv_act_scale_out)
  float combined = act_scale_in * inv_rms * inv_act_scale_out;

  // Pass 2: scalar fp32 throughout (no bf16 multiply chain). Gamma
  // loaded as bf16 from DRAM and cast once to fp32 per element; all
  // arithmetic is IEEE fp32 so the numpy reference can compute the
  // identical value bit-for-bit. Vec optimization (and bringing back
  // the bf16 math path) is a follow-up.
  for (int i = 0; i < kCols; i++) {
    float xf = (float)x[i];                  // int8 -> fp32 (exact)
    float gf = (float)gamma[i];              // bf16 -> fp32 (exact)
    float vf = xf * combined * gf;           // single fp32 chain
    y[i] = round_to_i8(vf);
  }

  event1();
}

} // extern "C"

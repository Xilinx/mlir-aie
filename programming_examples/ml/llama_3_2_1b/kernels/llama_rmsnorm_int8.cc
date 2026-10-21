//===- llama_rmsnorm_int8.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Production-shaped RMSNorm: int8 in, int8 out, bf16-internal. AIE2P DMA
// cannot type-convert, so the int8<->bf16 dequant/requant live inside the
// kernel rather than at the DMA boundary.
//
//   combined = act_scale_in * invsqrt(mean(x_f^2) + eps) * inv_act_scale_out
//   y_i8     = clamp(round(x_i8 * combined * gamma[i]), -128, 127)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

#ifndef LLAMA_RMSNORM_COLS
#define LLAMA_RMSNORM_COLS 2048
#endif

static constexpr float kEps = 1e-5f;
static constexpr int kVec =
    64; // one aie::mac lane group / one AIE2P 512-bit reg
// One 512-bit fp32 register.
static constexpr int kLanes = 16;

// 1/sqrt(a) from the Quake-III seed and two Newton steps, as aie_fp.sw_invsqrt
// mirrors it (not the correctly rounded value).
static inline float sw_invsqrt(float a) {
  int32_t bits = __builtin_bit_cast(int32_t, a);
  bits = (int32_t)0x5f3759df - (bits >> 1);
  float x = __builtin_bit_cast(float, bits);
  x = x * (1.5f - 0.5f * a * x * x);
  x = x * (1.5f - 0.5f * a * x * x);
  return x;
}

// Exact int32 sum-of-squares, kVec lanes at a time (an integer reduction, so
// its lane order cannot change the result).
static inline int32_t vec_sum_sq_i8(const int8_t *restrict x, int cols) {
  ::aie::accum<acc32, kVec> acc;
  acc.from_vector(::aie::zeros<int32, kVec>());
  const int chunks = cols / kVec;
  for (int c = 0; c < chunks; c++) {
    ::aie::vector<int8, kVec> xv = ::aie::load_v<kVec>(x + c * kVec);
    acc = ::aie::mac(acc, xv, xv);
  }
  return ::aie::reduce_add(acc.template to_vector<int32>());
}

// (x_i8 * combined) * gamma[i], kLanes at a time: exact widenings
// (int8->f32, bf16->f32) and two products each rounded once, as the scalar
// `*` chain does per lane.
static inline ::aie::vector<float, kLanes>
vec_normalize_lanes(const int8_t *restrict x, const bfloat16 *restrict gamma,
                    ::aie::vector<float, kLanes> combined_v) {
  ::aie::accum<accfloat, kLanes> g_acc;
  g_acc.from_vector(::aie::load_v<kLanes>(gamma));
  return vec_mul_rne(
      vec_mul_rne(::aie::to_float<float>(::aie::load_v<kLanes>(x)), combined_v),
      g_acc.template to_vector<float>());
}

// absmax of (x_i8 * pre) * gamma as an integer max of the magnitude bits.
static inline float vec_absmax(const int8_t *restrict x,
                               const bfloat16 *restrict gamma,
                               ::aie::vector<float, kLanes> pre_v, int cols) {
  ::aie::vector<int32, kLanes> mag = ::aie::zeros<int32, kLanes>();
  for (int c = 0; c < cols / kLanes; c++)
    mag = ::aie::max(
        mag, ::aie::bit_and(
                 int32(0x7fffffff),
                 vec_normalize_lanes(x + c * kLanes, gamma + c * kLanes, pre_v)
                     .template cast_to<int32>()));
  int32_t bits = ::aie::reduce_max(mag);
  float absmax = __builtin_bit_cast(float, bits);
  return absmax;
}

static inline void vec_requant(const int8_t *restrict x,
                               const bfloat16 *restrict gamma,
                               int8_t *restrict y,
                               ::aie::vector<float, kLanes> combined_v,
                               int cols) {
  for (int c = 0; c < cols / kLanes; c++)
    ::aie::store_v(y + c * kLanes,
                   vec_round_to_i8(vec_normalize_lanes(
                       x + c * kLanes, gamma + c * kLanes, combined_v)));
}

extern "C" {

// RMSNorm with a dynamic output scale: act_scale_in comes from x's own tail
// (x[kCols..kCols+4], written upstream by the residual add), and the output's
// fp32 scale, absmax/127, goes to y[kCols..kCols+4]. x and y must both be
// int8[kCols + 8].
void llama_rmsnorm_int8_dyn_acttail(int8_t *restrict x,
                                    bfloat16 *restrict gamma,
                                    int8_t *restrict y) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kVec == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 64");

  float act_scale_in;
  __builtin_memcpy(&act_scale_in, x + kCols, 4);

  int32_t sum_sq_i32 = vec_sum_sq_i8(x, kCols);
  float var = (float)sum_sq_i32 * act_scale_in * act_scale_in / (float)kCols;
  float inv_rms = sw_invsqrt(var + kEps);
  float pre = act_scale_in * inv_rms;

  float absmax =
      vec_absmax(x, gamma, ::aie::broadcast<float, kLanes>(pre), kCols);
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float scale_dyn = absmax * (1.0f / 127.0f);
  float inv_dyn = recip_rne(scale_dyn);
  float combined = pre * inv_dyn;
  vec_requant(x, gamma, y, ::aie::broadcast<float, kLanes>(combined), kCols);

  __builtin_memcpy(y + kCols, &scale_dyn, 4);
  int32_t zero = 0;
  __builtin_memcpy(y + kCols + 4, &zero, 4);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// x and gamma in one buffer: x int8[kCols + 8] at 0, gamma at kCols + 64
// (the GEMV array decode stages the two together).
void llama_rmsnorm_int8_dyn_acttail_packed(int8_t *restrict xg,
                                           int8_t *restrict y) {
  constexpr int kCols = LLAMA_RMSNORM_COLS;
  llama_rmsnorm_int8_dyn_acttail(
      xg, reinterpret_cast<bfloat16 *>(xg + kCols + 64), y);
}

} // extern "C"

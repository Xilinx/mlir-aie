//===- llama_rmsnorm_int8.cc -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RMSNorm: the fp32 residual in, int8 out with its dynamic scale. AIE2P DMA
// cannot type-convert, so the requant lives inside the kernel.
//
//   inv_rms  = invsqrt(mean(x^2) + eps)
//   combined = inv_rms * inv_act_scale_out
//   y_i8     = clamp(round(x * combined * gamma[i]), -128, 127)
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

#ifndef LLAMA_RMSNORM_COLS
#define LLAMA_RMSNORM_COLS 2048
#endif

static constexpr float kEps = 1e-5f;
// One 512-bit fp32 register.
static constexpr int kLanes = 16;

// 1/sqrt(a) from the Quake-III seed and two Newton steps, as
// llama_oracle.sw_invsqrt mirrors it (not the correctly rounded value).
static inline float sw_invsqrt(float a) {
  int32_t bits = __builtin_bit_cast(int32_t, a);
  bits = (int32_t)0x5f3759df - (bits >> 1);
  float x = __builtin_bit_cast(float, bits);
  x = scalar_mul_rne(
      x, 1.5f - scalar_mul_rne(scalar_mul_rne(scalar_mul_rne(0.5f, a), x), x));
  x = scalar_mul_rne(
      x, 1.5f - scalar_mul_rne(scalar_mul_rne(scalar_mul_rne(0.5f, a), x), x));
  return x;
}

// sum(x^2) in fp32: each lane sums its squares in order, then the lanes are
// added in order 0..15 as scalars, the order llama_oracle.sum_sq_f32 takes.
static inline float vec_sum_sq_f32(const float *restrict x, int cols) {
  ::aie::vector<float, kLanes> acc = ::aie::zeros<float, kLanes>();
  for (int c = 0; c < cols / kLanes; c++) {
    ::aie::vector<float, kLanes> xv = ::aie::load_v<kLanes>(x + c * kLanes);
    acc = ::aie::add(acc, vec_mul_rne(xv, xv));
  }
  alignas(64) float lanes[kLanes];
  ::aie::store_v(lanes, acc);
  float sum = lanes[0];
  for (int l = 1; l < kLanes; l++)
    sum += lanes[l];
  return sum;
}

// (x * combined) * gamma[i], kLanes at a time: an exact widening (bf16->f32)
// and two products each rounded once, as the scalar `*` chain does per lane.
static inline ::aie::vector<float, kLanes>
vec_normalize_lanes(const float *restrict x, const bfloat16 *restrict gamma,
                    ::aie::vector<float, kLanes> combined_v) {
  ::aie::accum<accfloat, kLanes> g_acc;
  g_acc.from_vector(::aie::load_v<kLanes>(gamma));
  return vec_mul_rne(vec_mul_rne(::aie::load_v<kLanes>(x), combined_v),
                     g_acc.template to_vector<float>());
}

// absmax of (x * pre) * gamma as an integer max of the magnitude bits.
static inline float vec_absmax(const float *restrict x,
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
  return __builtin_bit_cast(float, bits);
}

static inline void vec_requant(const float *restrict x,
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

// xg: the fp32 residual x[kCols], then gamma bf16[kCols] (the decode stages
// the two together). y: int8[kCols + 8], the normed row, then its fp32 scale
// absmax/127 and 4 zero bytes.
void llama_rmsnorm_f32_packed(int8_t *restrict xg, int8_t *restrict y) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kLanes == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 16");
  const float *x = reinterpret_cast<const float *>(xg);
  const bfloat16 *gamma = reinterpret_cast<const bfloat16 *>(xg + 4 * kCols);

  float var = scalar_mul_rne(vec_sum_sq_f32(x, kCols), 1.0f / kCols);
  float inv_rms = sw_invsqrt(var + kEps);

  float absmax =
      vec_absmax(x, gamma, ::aie::broadcast<float, kLanes>(inv_rms), kCols);
  // absmax is always a magnitude (sign bit clear), so its bit pattern orders
  // the same as its value: an unsigned int compare avoids __ltsf2.
  if ((uint32_t)__builtin_bit_cast(int32_t, absmax) <
      (uint32_t)__builtin_bit_cast(int32_t, 1e-12f))
    absmax = 1e-12f;
  float scale_dyn = scalar_mul_rne(absmax, 1.0f / 127.0f);
  float combined = scalar_mul_rne(inv_rms, recip_rne(scale_dyn));
  vec_requant(x, gamma, y, ::aie::broadcast<float, kLanes>(combined), kCols);

  __builtin_memcpy(y + kCols, &scale_dyn, 4);
  int32_t zero = 0;
  __builtin_memcpy(y + kCols + 4, &zero, 4);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// llama_rmsnorm_f32_packed, then x on to row: the residual the residual core
// adds the unit's output to.
void llama_rmsnorm_f32_packed_fwd(int8_t *restrict xg, int8_t *restrict y,
                                  int8_t *restrict row) {
  llama_rmsnorm_f32_packed(xg, y);
  for (int c = 0; c < 4 * LLAMA_RMSNORM_COLS / 64; c++)
    ::aie::store_v(row + c * 64, ::aie::load_v<64>(xg + c * 64));
}

} // extern "C"

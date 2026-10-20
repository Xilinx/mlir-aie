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
#include <string.h>

#include "llama_vec_fp.h"

#ifndef LLAMA_RMSNORM_COLS
#define LLAMA_RMSNORM_COLS 2048
#endif

static constexpr float kEps = 1e-5f;
static constexpr int kVec =
    64; // one aie::mac lane group / one AIE2P 512-bit reg
// One 512-bit fp32 register.
static constexpr int kLanes = 16;

// See vec_sw_recip in llama_vec_fp.h; this is its scalar form.
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
  float absmax;
  memcpy(&absmax, &bits, 4);
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

void llama_rmsnorm_int8(int8_t *restrict x, bfloat16 *restrict gamma,
                        int8_t *restrict y, float act_scale_in,
                        float inv_act_scale_out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kVec == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 64");

  // Pass 1: exact int32 sum-of-squares over int8 inputs. int8 * int8
  // = int16; 2048 lanes worst-case = 2048 * 16384 = 33M, fits in int32.
  int32_t sum_sq_i32 = vec_sum_sq_i8(x, kCols);

  // Fold act_scale_in into the variance: x_f^2 = act_scale_in^2 * x_i8^2.
  // 1/kCols (kCols=64) is exactly representable in fp32 as 2^-6, so
  // the Peano HW reciprocal happens to be IEEE-correct for this
  // specific case -- regular division works here.
  float var = (float)sum_sq_i32 * act_scale_in * act_scale_in / (float)kCols;
  float inv_rms = sw_invsqrt(var + kEps);

  // Combined per-output multiplier (single fp32 scalar):
  //   y_f = x_i8 * act_scale_in * inv_rms * gamma[i]
  //   y_i8 = round(y_f * inv_act_scale_out)
  float combined = act_scale_in * inv_rms * inv_act_scale_out;
  vec_requant(x, gamma, y, ::aie::broadcast<float, kLanes>(combined), kCols);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// Dynamic-output-scale variant: writes the fp32 dynamic scale (absmax/127)
// into y[kCols..kCols+4]. Recomputes the fp32 chain twice (find absmax,
// then requant) rather than buffering kCols fp32 between passes -- Peano
// AIE2P corrupts stack-allocated fp32 arrays kept live across loops.
// y must be int8[kCols + 8] (4 B scale + 4 B pad, 64 B-aligned).
void llama_rmsnorm_int8_dyn(int8_t *restrict x, bfloat16 *restrict gamma,
                            int8_t *restrict y, float act_scale_in) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kVec == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 64");

  int32_t sum_sq_i32 = vec_sum_sq_i8(x, kCols);
  float var = (float)sum_sq_i32 * act_scale_in * act_scale_in / (float)kCols;
  float inv_rms = sw_invsqrt(var + kEps);
  float pre = act_scale_in * inv_rms; // x_i8 * pre * gamma = y_f (pre-requant)

  float absmax =
      vec_absmax(x, gamma, ::aie::broadcast<float, kLanes>(pre), kCols);
  // Guard against all-zero rows (avoid div-by-zero); 1e-12 matches the
  // numpy reference's floor in cautious-eureka quant_act.
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float scale_dyn =
      absmax * (1.0f / 127.0f); // fp32 const; literal /127 is fine
  float inv_dyn = sw_recip(scale_dyn);
  float combined = pre * inv_dyn;
  vec_requant(x, gamma, y, ::aie::broadcast<float, kLanes>(combined), kCols);

  // Write the fp32 scale to the tail. memcpy avoids strict-aliasing UB.
  memcpy(y + kCols, &scale_dyn, 4);
  // Zero the 4 B pad so deterministic byte-compares pass.
  int32_t zero = 0;
  memcpy(y + kCols + 4, &zero, 4);

  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// Same as llama_rmsnorm_int8_dyn, but act_scale_in comes from x's own tail
// (x[kCols..kCols+4], written upstream by rescale-add) instead of an arg --
// the same acttail pattern gemm's ...gate_acttail uses. x and y must both
// be int8[kCols + 8].
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
  memcpy(&act_scale_in, x + kCols, 4);

  int32_t sum_sq_i32 = vec_sum_sq_i8(x, kCols);
  float var = (float)sum_sq_i32 * act_scale_in * act_scale_in / (float)kCols;
  float inv_rms = sw_invsqrt(var + kEps);
  float pre = act_scale_in * inv_rms;

  float absmax =
      vec_absmax(x, gamma, ::aie::broadcast<float, kLanes>(pre), kCols);
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float scale_dyn = absmax * (1.0f / 127.0f);
  float inv_dyn = sw_recip(scale_dyn);
  float combined = pre * inv_dyn;
  vec_requant(x, gamma, y, ::aie::broadcast<float, kLanes>(combined), kCols);

  memcpy(y + kCols, &scale_dyn, 4);
  int32_t zero = 0;
  memcpy(y + kCols + 4, &zero, 4);

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

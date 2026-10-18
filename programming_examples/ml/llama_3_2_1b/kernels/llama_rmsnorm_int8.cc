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

#ifndef LLAMA_RMSNORM_COLS
#define LLAMA_RMSNORM_COLS 2048
#endif

static constexpr float kEps = 1e-5f;
static constexpr int kVec =
    64; // one aie::mac lane group / one AIE2P 512-bit reg
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

// Software fp32 reciprocal (see llama_flowkv.cc for why Peano's `/` needs
// this workaround).
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

// Exact int32 sum-of-squares, kVec lanes at a time (see vec_normalize_chunk
// for why vectorizing preserves bit-exactness here).
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

// (x_i8 * combined) * gamma[i], kVec lanes at a time. Bit-exactness with
// the scalar/numpy reference holds throughout this file because every
// vectorized step is either an exact widening (int8->f32, bf16->f32), an
// order-independent reduction (integer add, float max), or -- here -- an
// elementwise map that preserves the reference's left-to-right fp32
// chain per lane; only a true reduction's lane order could change
// rounding, and none of the fp32 math below is one.
static inline ::aie::vector<float, kVec>
vec_normalize_chunk(const int8_t *restrict x, const bfloat16 *restrict gamma,
                    ::aie::vector<float, kVec> combined_v) {
  ::aie::vector<int8, kVec> xv = ::aie::load_v<kVec>(x);
  ::aie::vector<bfloat16, kVec> gv = ::aie::load_v<kVec>(gamma);
  ::aie::vector<float, kVec> xf = ::aie::to_float<float>(xv);
  ::aie::accum<accfloat, kVec> g_acc;
  g_acc.from_vector(gv);
  ::aie::vector<float, kVec> gf = g_acc.template to_vector<float>();
  ::aie::vector<float, kVec> t =
      ::aie::mul(xf, combined_v).template to_vector<float>();
  return ::aie::mul(t, gf).template to_vector<float>();
}

extern "C" {

void llama_rmsnorm_int8(int8_t *restrict x, bfloat16 *restrict gamma,
                        int8_t *restrict y, float act_scale_in,
                        float inv_act_scale_out) {
  event0();

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kVec == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 64");
  constexpr int kChunks = kCols / kVec;

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
  ::aie::vector<float, kVec> combined_v =
      ::aie::broadcast<float, kVec>(combined);

  // Pass 2: only the final round_to_i8 stays scalar.
  for (int c = 0; c < kChunks; c++) {
    ::aie::vector<float, kVec> vf =
        vec_normalize_chunk(x + c * kVec, gamma + c * kVec, combined_v);
    for (int i = 0; i < kVec; i++)
      y[c * kVec + i] = round_to_i8(vf[i]);
  }

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

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kVec == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 64");
  constexpr int kChunks = kCols / kVec;

  int32_t sum_sq_i32 = vec_sum_sq_i8(x, kCols);
  float var = (float)sum_sq_i32 * act_scale_in * act_scale_in / (float)kCols;
  float inv_rms = sw_invsqrt(var + kEps);
  float pre = act_scale_in * inv_rms; // x_i8 * pre * gamma = y_f (pre-requant)
  ::aie::vector<float, kVec> pre_v = ::aie::broadcast<float, kVec>(pre);

  // Pass A: scan fp32 y_f, find absmax. No fp32 array kept across loops --
  // each chunk's vf vector is consumed within the same iteration.
  float absmax = 0.0f;
  for (int c = 0; c < kChunks; c++) {
    ::aie::vector<float, kVec> vf =
        vec_normalize_chunk(x + c * kVec, gamma + c * kVec, pre_v);
    float chunk_max = ::aie::reduce_max(::aie::abs(vf));
    if (chunk_max > absmax)
      absmax = chunk_max;
  }

  // Guard against all-zero rows (avoid div-by-zero); 1e-12 matches the
  // numpy reference's floor in cautious-eureka quant_act.
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float scale_dyn =
      absmax * (1.0f / 127.0f); // fp32 const; literal /127 is fine
  float inv_dyn = sw_recip(scale_dyn);
  float combined = pre * inv_dyn;
  ::aie::vector<float, kVec> combined_v =
      ::aie::broadcast<float, kVec>(combined);

  // Pass B: recompute the same chain, round to int8 with the dynamic scale.
  for (int c = 0; c < kChunks; c++) {
    ::aie::vector<float, kVec> vf =
        vec_normalize_chunk(x + c * kVec, gamma + c * kVec, combined_v);
    for (int i = 0; i < kVec; i++)
      y[c * kVec + i] = round_to_i8(vf[i]);
  }

  // Write the fp32 scale to the tail. memcpy avoids strict-aliasing UB.
  memcpy(y + kCols, &scale_dyn, 4);
  // Zero the 4 B pad so deterministic byte-compares pass.
  int32_t zero = 0;
  memcpy(y + kCols + 4, &zero, 4);

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

  constexpr int kCols = LLAMA_RMSNORM_COLS;
  static_assert(kCols % kVec == 0,
                "LLAMA_RMSNORM_COLS must be a multiple of 64");
  constexpr int kChunks = kCols / kVec;

  float act_scale_in;
  memcpy(&act_scale_in, x + kCols, 4);

  int32_t sum_sq_i32 = vec_sum_sq_i8(x, kCols);
  float var = (float)sum_sq_i32 * act_scale_in * act_scale_in / (float)kCols;
  float inv_rms = sw_invsqrt(var + kEps);
  float pre = act_scale_in * inv_rms;
  ::aie::vector<float, kVec> pre_v = ::aie::broadcast<float, kVec>(pre);

  float absmax = 0.0f;
  for (int c = 0; c < kChunks; c++) {
    ::aie::vector<float, kVec> vf =
        vec_normalize_chunk(x + c * kVec, gamma + c * kVec, pre_v);
    float chunk_max = ::aie::reduce_max(::aie::abs(vf));
    if (chunk_max > absmax)
      absmax = chunk_max;
  }

  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float scale_dyn = absmax * (1.0f / 127.0f);
  float inv_dyn = sw_recip(scale_dyn);
  float combined = pre * inv_dyn;
  ::aie::vector<float, kVec> combined_v =
      ::aie::broadcast<float, kVec>(combined);

  for (int c = 0; c < kChunks; c++) {
    ::aie::vector<float, kVec> vf =
        vec_normalize_chunk(x + c * kVec, gamma + c * kVec, combined_v);
    for (int i = 0; i < kVec; i++)
      y[c * kVec + i] = round_to_i8(vf[i]);
  }

  memcpy(y + kCols, &scale_dyn, 4);
  int32_t zero = 0;
  memcpy(y + kCols + 4, &zero, 4);

  event1();
}

} // extern "C"

//===---  exp_lut.h - get exponential values from loopup tables ---===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is the implementation of getting exponential values for a bfloat16
// vector from exponential lookup tables.
//===----------------------------------------------------------------------===//
#ifndef __LUT_BASED_OPS_H__
#define __LUT_BASED_OPS_H__

#include "aie_api/aie.hpp"

alignas(aie::vector_decl_align) extern int16 exp_ilut_ab[512];
alignas(aie::vector_decl_align) extern int16 exp_ilut_cd[512];
alignas(aie::vector_decl_align) extern int16 exp_flut_ab[512];
alignas(aie::vector_decl_align) extern int16 exp_flut_cd[512];
alignas(aie::vector_decl_align) extern unsigned char m_inv_lut[128];

// Clamp to the LUT's supported range before Q8 conversion can wrap.
static constexpr float EXP_BF16_CLAMP = 88.0f;

// exp(x) = exp(int(x)) * exp(frac(x)), from a 256-entry table each: the
// aie::parallel_lookup of the Q8 input (steps 8 and 0) written out. Its byte
// offsets are floor(4x) and floor(1024x) within the 1 KiB tables; vfloor
// floors whatever crRnd is, where parallel_lookup's accumulator shift needed
// crRnd set to floor, and saving, setting and restoring it on every call kept
// the calling loops from pipelining. The integer offset's two low bits select
// the same entry, so both are masked to 0x3FC.
__attribute__((always_inline)) v16accfloat getExpBf16(v16bfloat16 x) {
  aie::vector<bfloat16, 16> xc =
      aie::max(aie::min(aie::vector<bfloat16, 16>(x),
                        aie::broadcast<bfloat16, 16>((bfloat16)EXP_BF16_CLAMP)),
               aie::broadcast<bfloat16, 16>((bfloat16)-EXP_BF16_CLAMP));
  v16int32 index_i = ::band(bfloat16_to_int(xc, 2), broadcast_s32(0x3FC));
  v16int32 index_f = ::band(bfloat16_to_int(xc, 10), broadcast_s32(0x3FC));

  v64int8 i0, i1, f0, f1;
  load_lut_2x_int8(exp_ilut_ab, exp_ilut_cd, index_i, i0, i1);
  load_lut_2x_int8(exp_flut_ab, exp_flut_cd, index_f, f0, f1);
  aie::vector<bfloat16, 32> i_val = (v32bfloat16)::shuffle(i0, i1, T16_16x4_lo);
  v32bfloat16 f_val = (v32bfloat16)::shuffle(f0, f1, T16_16x4_lo);
  i_val.insert<16>(1, aie::zeros<bfloat16, 16>());
  return mul_elem_16_2(i_val, f_val);
}

__attribute__((always_inline)) bfloat16 getInvBf16(float x) {
  unsigned int *B_x;
  unsigned int exp_mask = 0x7F800000;
  unsigned int mantissa_mask = 0x007FFFFF;
  unsigned int mantissa_Q = 0x00008000;
  unsigned char exponent, mantissa;
  unsigned inv_exponent;
  unsigned short inv_x_val;
  unsigned int B_Q;
  bfloat16 *inv_x;
  B_x = (unsigned int *)&x;
  B_Q = *B_x + mantissa_Q;
  exponent = (B_Q & exp_mask) >> 23;
  mantissa = (B_Q & mantissa_mask) >> 16;
  inv_exponent = (mantissa == 0) + (253 - exponent);
  inv_x_val = (inv_exponent << 7) + m_inv_lut[mantissa];
  inv_x = (bfloat16 *)&inv_x_val;
  return *inv_x;
}

extern float tanh_lut_ab[];
extern float tanh_lut_cd[];

// aie::linear_approx<bfloat16, aie::lut<4, float, bfloat16>> with step_bits
// -2 and bias 16, written out: 32 segments of 0.25 over [-4, 4), each
// offset + slope * x. The object form is rebuilt on every call, and its
// scratchpad member makes it escape, so each call stored the whole object to
// the stack and read the input back through it.
inline __attribute__((always_inline)) v16bfloat16
getTanhBf16(v16bfloat16 vInput) {
  // Byte offset of the segment: floor(x * 4) entries of 16 bytes, relative to
  // the middle of the table. x is clamped to the table's range first; the end
  // segments have slope 0, so the result is unchanged, and +-inf no longer
  // makes 0 * inf.
  constexpr int bias_bytes = 16 << 4;
  const float *lut_ab = tanh_lut_ab + bias_bytes / sizeof(float);
  const float *lut_cd = tanh_lut_cd + bias_bytes / sizeof(float);
  aie::vector<bfloat16, 16> xc = aie::max(
      aie::min(aie::vector<bfloat16, 16>(vInput), bfloat16(4.0f - 1.0f / 64)),
      bfloat16(-4.0f));
  v16int32 index = bfloat16_to_int(xc, 6);

  v32bfloat16 coeff0, coeff1;
  load_lut_2x_float(lut_ab, lut_cd, index, coeff0, coeff1);
  v16accfloat offset = (v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi);
  v32bfloat16 slope = ::shuffle(coeff0, coeff1, T16_16x4_lo);
  aie::vector<bfloat16, 32> x = aie::zeros<bfloat16, 32>();
  x.insert<16>(1, xc);

  aie::accum<accfloat, 16> result = mac_elem_16_2(slope, x, offset);
  return (v16bfloat16)result.to_vector<bfloat16>();
}
// Applies f, a function of one 16-lane vector that reads a table, to n
// elements, a multiple of 16. The table reads are ordered against every other
// load and store, so a loop that loads one vector, looks it up and stores it
// runs them one after another. K vectors per trip instead, and all K stores
// after all K lookups. With Prefetch each trip's input is loaded by the one
// before it (the last trip reloads its own), which pays where f does more
// than the lookup (gelu, silu) and costs where it does little else (tanh).
template <int K = 4, bool Prefetch = true, typename F>
inline __attribute__((always_inline)) void
lut_map_bf16(const bfloat16 *restrict in, bfloat16 *restrict out, int n, F f) {
  using V = aie::vector<bfloat16, 16>;
  auto it_out = aie::begin_restrict_vector<16>(out);
  const int trips = n / (16 * K);
  if (trips > 0) {
    V next[K];
    if constexpr (Prefetch)
      for (int j = 0; j < K; j++)
        next[j] = aie::load_v<16>(in + 16 * j);
    for (int i = 0; i < trips; i++) {
      V x[K], y[K];
      if constexpr (Prefetch) {
        for (int j = 0; j < K; j++)
          x[j] = next[j];
        const bfloat16 *p = in + (i + 1 < trips ? i + 1 : i) * 16 * K;
        for (int j = 0; j < K; j++)
          next[j] = aie::load_v<16>(p + 16 * j);
      } else {
        for (int j = 0; j < K; j++)
          x[j] = aie::load_v<16>(in + i * 16 * K + 16 * j);
      }
      for (int j = 0; j < K; j++)
        y[j] = f(x[j]);
      for (int j = 0; j < K; j++)
        *it_out++ = y[j];
    }
  }
  auto it_in = aie::begin_restrict_vector<16>(in + trips * 16 * K);
  for (int i = 0; i < n % (16 * K); i += 16)
    *it_out++ = f(*it_in++);
}
#endif //__LUT_BASED_OPS_H__
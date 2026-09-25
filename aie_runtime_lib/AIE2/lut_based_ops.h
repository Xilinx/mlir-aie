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

__attribute__((always_inline)) v16accfloat getExpBf16(v16bfloat16 x) {
  bfloat16 __aie_dm_resource_a *ilut_ab =
      (bfloat16 __aie_dm_resource_a *)exp_ilut_ab;
  bfloat16 __aie_dm_resource_b *ilut_cd =
      (bfloat16 __aie_dm_resource_b *)exp_ilut_cd;
  bfloat16 __aie_dm_resource_a *flut_ab =
      (bfloat16 __aie_dm_resource_a *)exp_flut_ab;
  bfloat16 __aie_dm_resource_b *flut_cd =
      (bfloat16 __aie_dm_resource_b *)exp_flut_cd;

  using lut_type = aie::lut<4, bfloat16, bfloat16>;
  const int LUT_elems = 256;
  const int step_i = 8;
  const int step_f = 0;

  lut_type lut_i(LUT_elems, ilut_ab, ilut_cd);
  lut_type lut_f(LUT_elems, flut_ab, flut_cd);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_i(lut_i, step_i);
  aie::parallel_lookup<uint16, lut_type, aie::lut_oor_policy::truncate>
      lookup_f(lut_f, step_f);

  aie::vector<bfloat16, 16> I_val_vec, F_val_vec;
  aie::accum<accfloat, 16> exp_val;
  aie::vector<bfloat16, 16> input_bf16 = x;

  // -max(-x, -c) also saturates +inf, unlike min(x, c) on AIE2P.
  input_bf16 = aie::neg(
      aie::max(aie::neg(input_bf16),
               aie::broadcast<bfloat16, 16>((bfloat16)-EXP_BF16_CLAMP)));
  input_bf16 = aie::max(
      input_bf16, aie::broadcast<bfloat16, 16>((bfloat16)-EXP_BF16_CLAMP));

  // position of output decimal point = 8, making input become 8 bits, and for
  // LUT_elems = 256 lookup. aie::vector<int16, 16>
  // input=aie::to_fixed<int16>(input_bf16,8);
  aie::vector<int16, 32> input0 = v32int16(bfloat16_to_int(input_bf16, 8));
  aie::vector<int16, 16> input = aie::filter_even(input0);

  // Lookup indices require floor rounding (aie_api CRVO-4425).
  aie::rounding_mode saved_rnd = aie::tile::current().get_rounding();
  aie::tile::current().set_rounding(aie::rounding_mode::floor);
  I_val_vec = lookup_i.fetch(input.cast_to<uint16>());
  F_val_vec = lookup_f.fetch(input.cast_to<uint16>());
  aie::tile::current().set_rounding(saved_rnd);

  exp_val = aie::mul(I_val_vec, F_val_vec);
  return v16accfloat(exp_val);
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
  // the middle of the table and clamped to its 32 entries.
  constexpr int bias_bytes = 16 << 4;
  const float *lut_ab = tanh_lut_ab + bias_bytes / sizeof(float);
  const float *lut_cd = tanh_lut_cd + bias_bytes / sizeof(float);
  v16int32 index = bfloat16_to_int(vInput, 6);
  index = ::max(index, aie::broadcast<int32, 16>(-bias_bytes));
  index = ::min(index, aie::broadcast<int32, 16>((32 << 4) - 1 - bias_bytes));

  v32bfloat16 coeff0, coeff1;
  load_lut_2x_float(lut_ab, lut_cd, index, coeff0, coeff1);
  v16accfloat offset = (v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi);
  v32bfloat16 slope = ::shuffle(coeff0, coeff1, T16_16x4_lo);
  aie::vector<bfloat16, 32> x = aie::zeros<bfloat16, 32>();
  x.insert<16>(1, aie::vector<bfloat16, 16>(vInput));

  aie::accum<accfloat, 16> result = mac_elem_16_2(slope, x, offset);
  return (v16bfloat16)result.to_vector<bfloat16>();
}
#endif //__LUT_BASED_OPS_H__
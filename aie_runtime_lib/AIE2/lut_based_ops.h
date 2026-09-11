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

// getExpBf16 indexes its tables through a Q8 fixed-point int16, so the inputs
// it can address at all are (-128, 128); past that the conversion wraps and
// the entry fetched has nothing to do with exp(x) (x = 200 measured as
// 4.8e-25, x = -200 as 2.1e+24). The tables themselves stop earlier: entry 88
// holds bf16(exp(88)) = 1.6549e+38 and is the largest value they carry, and
// exp(-88) has already underflowed bf16 to 0. Clamping the input here is
// therefore exact for every x the kernel could otherwise answer correctly,
// and turns the wrap into the saturation callers expect. exp2f_vec takes the
// same approach with its min_x.
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

  // Saturate to the addressable domain (see EXP_BF16_CLAMP) before the Q8
  // conversion below, which would otherwise wrap.
  input_bf16 = aie::min(input_bf16,
                        aie::broadcast<bfloat16, 16>((bfloat16)EXP_BF16_CLAMP));
  input_bf16 = aie::max(
      input_bf16, aie::broadcast<bfloat16, 16>((bfloat16)-EXP_BF16_CLAMP));

  // position of output decimal point = 8, making input become 8 bits, and for
  // LUT_elems = 256 lookup. aie::vector<int16, 16>
  // input=aie::to_fixed<int16>(input_bf16,8);
  aie::vector<int16, 32> input0 = v32int16(bfloat16_to_int(input_bf16, 8));
  aie::vector<int16, 16> input = aie::filter_even(input0);

  // The parallel_lookup fetch()es above convert a fixed-point accumulator to
  // a table index using whatever core-wide rounding mode is active. A caller
  // that sets conv_even before this loop (to narrow its own accumulator to
  // bf16 -- see KernelContract.rounding_mode) corrupts the index for any Q8
  // fraction >= 224/256, returning exp(frac + 1) instead of exp(frac).
  // Bracketing the fetches in floor (the core's boot default, and the mode
  // these tables were authored against) avoids that; HW-verified on Phoenix.
  // aie_api carries a FIXME for this (detail/aie2/parallel_lookup.hpp,
  // CRVO-4425) that would make the bracket unnecessary once fixed upstream.
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

inline __attribute__((always_inline)) v16bfloat16
getTanhBf16(v16bfloat16 vInput) {
  aie::vector<bfloat16, 16> input = vInput;

  int step_bits = -2;
  int bias = 16;
  int data_size = 16;
  int LUT_elems = 32;
  int shift_offset = 0; // unused

  using lut_type = aie::lut<4, float, bfloat16>;

  lut_type test_lut(LUT_elems, (bfloat16 *)tanh_lut_ab,
                    (bfloat16 *)tanh_lut_cd);

  aie::linear_approx<bfloat16, lut_type> lin_aprox(test_lut, step_bits, bias,
                                                   shift_offset);

  aie::vector<bfloat16, 16> output =
      lin_aprox.compute(input).to_vector<bfloat16>();

  return (v16bfloat16)output;
}
#endif //__LUT_BASED_OPS_H__
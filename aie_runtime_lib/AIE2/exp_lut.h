//===---  exp_lut.h - get exponential values from loopup tables ---===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//
//===----------------------------------------------------------------------===//
// This is the implementation of getting exponential values for a bfloat16
// vector from exponential lookup tables.
//===----------------------------------------------------------------------===//
#ifndef __EXP_LUT_H__
#define __EXP_LUT_H__

#include "aie_api/aie.hpp"

alignas(aie::vector_decl_align) extern int16 softmax_ilut_ab[512];
alignas(aie::vector_decl_align) extern int16 softmax_ilut_cd[512];
alignas(aie::vector_decl_align) extern int16 softmax_flut_ab[512];
alignas(aie::vector_decl_align) extern int16 softmax_flut_cd[512];

__attribute__((always_inline)) v16accfloat getExpBf16(v16bfloat16 x) {
  bfloat16 __aie_dm_resource_a *ilut_ab =
      (bfloat16 __aie_dm_resource_a *)softmax_ilut_ab;
  bfloat16 __aie_dm_resource_b *ilut_cd =
      (bfloat16 __aie_dm_resource_b *)softmax_ilut_cd;
  bfloat16 __aie_dm_resource_a *flut_ab =
      (bfloat16 __aie_dm_resource_a *)softmax_flut_ab;
  bfloat16 __aie_dm_resource_b *flut_cd =
      (bfloat16 __aie_dm_resource_b *)softmax_flut_cd;

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

  // position of output decimal point = 8, making input become 8 bits, and for
  // LUT_elems = 256 lookup. aie::vector<int16, 16>
  // input=aie::to_fixed<int16>(input_bf16,8);
  aie::vector<int16, 32> input0 = v32int16(bfloat16_to_int(input_bf16, 8));
  aie::vector<int16, 16> input = aie::filter_even(input0);

  I_val_vec = lookup_i.fetch(input.cast_to<uint16>());
  F_val_vec = lookup_f.fetch(input.cast_to<uint16>());
  exp_val = aie::mul(I_val_vec, F_val_vec);
  return v16accfloat(exp_val);
}
#endif //__EXP_LUT_H__

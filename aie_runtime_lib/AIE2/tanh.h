//===-  tanh.h - get hyperbolic tangent values based on linear approximation
//-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//
//===----------------------------------------------------------------------===//
// This is the implementation of compute hyperbolic tangent values based on
// linear approximation
//===----------------------------------------------------------------------===//

#ifndef __TANH__
#define __TANH__

#include "aie_api/aie.hpp"
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

extern float chess_storage(% chess_alignof(v32int8)) tanh_lut_ab[2048];
extern float chess_storage(% chess_alignof(v32int8)) tanh_lut_cd[2048];

aie::vector<bfloat16, 16> __attribute__((always_inline))
linear_approx_mode_6(aie::vector<bfloat16, 16> input, bfloat16 *LUTab,
                     bfloat16 *LUTcd, int bias, int LUT_elems,
                     int remainder_bits, int shift_offset) {
  using lut_type = aie::lut<4, float, bfloat16>;

  lut_type test_lut(LUT_elems, LUTab, LUTcd);

  aie::linear_approx<bfloat16, lut_type> lin_aprox(test_lut, remainder_bits,
                                                   bias, shift_offset);

  return lin_aprox.compute(input).to_vector<bfloat16>();
}

v16bfloat16 __attribute__((noinline)) getTanhBf16(v16bfloat16 window_input) {
  aie::vector<bfloat16, 16> input = window_input;

  int step_bits = -2;
  int bias = 16;
  int data_size = 16;
  int LUT_elems = 32;
  int shift_offset = 0; // unused

  aie::vector<bfloat16, 16> output = linear_approx_mode_6(
      input, (bfloat16 *)tanh_lut_ab, (bfloat16 *)tanh_lut_cd, bias, LUT_elems,
      step_bits, shift_offset);
  return (v16bfloat16)output;
}

#endif //__TANH__

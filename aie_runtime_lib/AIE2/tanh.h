//===-  tanh.h - get hyperbolic tangent values based on linear approximation
//-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
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

#endif //__TANH__

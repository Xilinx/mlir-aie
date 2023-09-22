//===-  erf.h - get error function values for bfloat16 data type
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
// This is the implementation of compute error function values for bfloat16 type
//===----------------------------------------------------------------------===//

#ifndef __ERF_H__
#define __ERF_H__

#include "aie_api/aie.hpp"

inline __attribute__((always_inline)) v32bfloat16 getErfBf16(v32bfloat16 in) {
  // Approximate the erf() using Maclaurin expansion, with x^1, x^3, and x^5
  // degrees generate x^1 - x^5
  constexpr bfloat16 Q1 = 1.128379167095512;
  constexpr bfloat16 Q3 = -0.3761263890318375;
  constexpr bfloat16 Q5 = 0.11283791670955;
  constexpr bfloat16 one = 1.0;
  constexpr bfloat16 minus_one = -1.0;
  aie::vector<bfloat16, 32> x1 = in;
  aie::accum<accfloat, 32> x2_float = aie::mul_square(x1);
  aie::vector<bfloat16, 32> x2 = x2_float.to_vector<bfloat16>();
  aie::accum<accfloat, 32> x3_float = aie::mul(x2, x1);
  aie::vector<bfloat16, 32> x3 = x3_float.to_vector<bfloat16>();
  aie::accum<accfloat, 32> x4_float = aie::mul_square(x2);
  aie::vector<bfloat16, 32> x4 = x4_float.to_vector<bfloat16>();
  aie::accum<accfloat, 32> x5_float = aie::mul(x4, x1);
  aie::vector<bfloat16, 32> x5 = x5_float.to_vector<bfloat16>();

  // accumulate Qi * x^i
  aie::accum<accfloat, 32> acc = aie::mul(x1, Q1);
  acc = aie::mac(acc, x3, Q3);
  acc = aie::mac(acc, x5, Q5);

  // process overflows
  aie::vector<bfloat16, 32> res = acc.to_vector<bfloat16>();
  aie::vector<bfloat16, 32> min_vec = aie::min(res, one);
  aie::vector<bfloat16, 32> vec_out = aie::max(min_vec, minus_one);
  return (v32bfloat16)vec_out;
}

inline __attribute__((always_inline)) v16bfloat16 getErfBf16(v16bfloat16 in) {
  constexpr bfloat16 Q1 = 1.128379167095512;
  constexpr bfloat16 Q3 = -0.3761263890318375;
  constexpr bfloat16 Q5 = 0.11283791670955;
  constexpr bfloat16 one = 1.0;
  constexpr bfloat16 minus_one = -1.0;
  aie::vector<bfloat16, 16> x1 = in;
  aie::accum<accfloat, 16> x2_float = aie::mul_square(x1);
  aie::vector<bfloat16, 16> x2 = x2_float.to_vector<bfloat16>();
  aie::accum<accfloat, 16> x3_float = aie::mul(x2, x1);
  aie::vector<bfloat16, 16> x3 = x3_float.to_vector<bfloat16>();
  aie::accum<accfloat, 16> x4_float = aie::mul_square(x2);
  aie::vector<bfloat16, 16> x4 = x4_float.to_vector<bfloat16>();
  aie::accum<accfloat, 16> x5_float = aie::mul(x4, x1);
  aie::vector<bfloat16, 16> x5 = x5_float.to_vector<bfloat16>();

  // accumulate Qi * x^i
  aie::accum<accfloat, 16> acc = aie::mul(x1, Q1);
  acc = aie::mac(acc, x3, Q3);
  acc = aie::mac(acc, x5, Q5);

  // process overflows
  aie::vector<bfloat16, 16> res = acc.to_vector<bfloat16>();
  aie::vector<bfloat16, 16> min_vec = aie::min(res, one);
  aie::vector<bfloat16, 16> vec_out = aie::max(min_vec, minus_one);
  return (v16bfloat16)vec_out;
}
#endif //__ERF_H__

//===-  vec_math.h -====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef VEC_MATH_H
#define VEC_MATH_H

#include "aie_api/aie.hpp"

// This function implements the computation of square root by the fast inverse
// square root implementation from Quake III Arena. We can solve square root
// just by multiplying the inverse square to the original number.
// float Q_rsqrt( float number )
//{
//	long i;
//	float x2, y;
//	const float threehalfs = 1.5F;
//
//	x2 = number * 0.5F;
//	y  = number;
//	i  = * ( long * ) &y;
//	i  = 0x5f3759df - ( i >> 1 );
//	y  = * ( float * ) &i;
//	y  = y * ( threehalfs - ( x2 * y * y ) );
//
//	return y;
//}
inline __attribute__((always_inline)) v32bfloat16 getRsqrtBf16(v32bfloat16 in) {
  aie::vector<bfloat16, 32> x = in;
  aie::accum<accfloat, 32> x2 =
      aie::mul(x, bfloat16(0.5f)); // x2 = number * 0.5F;
  const aie::vector<int16, 32> magic = aie::broadcast<int16, 32>(0x5f37);
  aie::vector<bfloat16, 32> y =
      aie::sub(magic, aie::downshift(x.cast_to<int16>(), 1))
          .cast_to<bfloat16>();

  x2 = aie::mul(x2.to_vector<bfloat16>(), y); // x2 * y
  x2 = aie::mul(x2.to_vector<bfloat16>(), y); // x2 * y * y
  const aie::vector<bfloat16, 32> threeHalfs =
      aie::broadcast<bfloat16, 32>(1.5f);
  // threehalfs - (x2 * y * y)
  auto t = aie::sub(threeHalfs, x2.to_vector<bfloat16>());
  x2 = aie::mul(t, y); // y * (threehalfs - (x2 * y * y) )
  aie::vector<bfloat16, 32> out = x2.to_vector<bfloat16>();
  return (v32bfloat16)out;
}

inline __attribute__((always_inline)) v32bfloat16 getSqrtBf16(v32bfloat16 in) {
  aie::vector<bfloat16, 32> x = in;
  aie::vector<bfloat16, 32> rsqrtX = getRsqrtBf16(in);
  aie::accum<accfloat, 32> sqrtX = aie::mul(x, rsqrtX);
  aie::vector<bfloat16, 32> out = sqrtX.to_vector<bfloat16>();
  return (v32bfloat16)out;
}

inline __attribute__((always_inline)) v16bfloat16 getRsqrtBf16(v16bfloat16 in) {
  aie::vector<bfloat16, 16> x = in;
  aie::accum<accfloat, 16> x2 =
      aie::mul(x, bfloat16(0.5f)); // x2 = number * 0.5F;
  const aie::vector<int16, 16> magic = aie::broadcast<int16, 16>(0x5f37);
  aie::vector<bfloat16, 16> y =
      aie::sub(magic, aie::downshift(x.cast_to<int16>(), 1))
          .cast_to<bfloat16>();

  x2 = aie::mul(x2.to_vector<bfloat16>(), y); // x2 * y
  x2 = aie::mul(x2.to_vector<bfloat16>(), y); // x2 * y * y
  const aie::vector<bfloat16, 16> threeHalfs =
      aie::broadcast<bfloat16, 16>(1.5f);
  // threehalfs - (x2 * y * y)
  auto t = aie::sub(threeHalfs, x2.to_vector<bfloat16>());
  x2 = aie::mul(t, y); // y * (threehalfs - (x2 * y * y) )
  aie::vector<bfloat16, 16> out = x2.to_vector<bfloat16>();
  return (v16bfloat16)out;
}

inline __attribute__((always_inline)) v16bfloat16 getSqrtBf16(v16bfloat16 in) {
  aie::vector<bfloat16, 16> x = in;
  aie::vector<bfloat16, 16> rsqrtX = getRsqrtBf16(in);
  aie::accum<accfloat, 16> sqrtX = aie::mul(x, rsqrtX);
  aie::vector<bfloat16, 16> out = sqrtX.to_vector<bfloat16>();
  return (v16bfloat16)out;
}

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

inline __attribute__((always_inline)) v32bfloat16 getAbs(v32bfloat16 in) {
  aie::vector<bfloat16, 32> x = in;
  aie::vector<bfloat16, 32> absX = aie::abs(x);
  return (v32bfloat16)absX;
}

inline __attribute__((always_inline)) v16bfloat16 getAbs(v16bfloat16 in) {
  aie::vector<bfloat16, 16> x = in;
  aie::vector<bfloat16, 16> absX = aie::abs(x);
  return (v16bfloat16)absX;
}

inline __attribute__((always_inline)) v16float getAbs(v16float in) {
  aie::vector<float, 16> x = in;
  aie::vector<float, 16> absX = aie::abs(x);
  return (v16float)absX;
}

inline __attribute__((always_inline)) v16int32 getAbs(v16int32 in) {
  aie::vector<int32, 16> x = in;
  aie::vector<int32, 16> absX = aie::abs(x);
  return (v16int32)absX;
}

inline __attribute__((always_inline)) v32int16 getAbs(v32int16 in) {
  aie::vector<int16, 32> x = in;
  aie::vector<int16, 32> absX = aie::abs(x);
  return (v32int16)absX;
}

inline __attribute__((always_inline)) v64int8 getAbs(v64int8 in) {
  aie::vector<int8, 64> x = in;
  aie::vector<int8, 64> absX = aie::abs(x);
  return (v64int8)absX;
}
#endif // VEC_MATH_H

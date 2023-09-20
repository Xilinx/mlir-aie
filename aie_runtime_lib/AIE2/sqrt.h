//===-  sqrt.h - get square root values for bfloat16 data type
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
// This is the implementation of compute square root values for bfloat16 type
//===----------------------------------------------------------------------===//

#ifndef __SQRT_H__
#define __SQRT_H__

#include "aie_api/aie.hpp"

__attribute__((always_inline)) v32bfloat16 getSqrtBf16(v32bfloat16 in) {
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
  x2 = aie::mul(x2.to_vector<bfloat16>(), x);
  aie::vector<bfloat16, 32> out = x2.to_vector<bfloat16>();
  return (v32bfloat16)out;
}

__attribute__((always_inline)) v16bfloat16 getSqrtBf16(v16bfloat16 in) {
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
  x2 = aie::mul(x2.to_vector<bfloat16>(), x);
  aie::vector<bfloat16, 16> out = x2.to_vector<bfloat16>();
  return (v16bfloat16)out;
}

#endif //__SQRT_H__

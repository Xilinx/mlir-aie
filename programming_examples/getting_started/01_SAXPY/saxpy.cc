//===- saxpy.cc --------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

extern "C" {
void saxpy(bfloat16 *restrict x, bfloat16 *restrict y, bfloat16 *restrict z) {
  event0();
  ::aie::vector<bfloat16, 64> a_v = ::aie::broadcast<bfloat16, 64>(3.f);
#pragma clang loop min_iteration_count(4)
  for (int i = 0; i < 4096; i += 64) {
    ::aie::vector<bfloat16, 64> x_v = ::aie::load_v<64>(x);
    x += 64;
    ::aie::vector<bfloat16, 64> y_v = ::aie::load_v<64>(y);
    y += 64;
    ::aie::accum<accfloat, 64> ax_v = ::aie::mul(x_v, a_v);
    ::aie::accum<accfloat, 64> z_v = ::aie::add(ax_v, y_v);
    ::aie::vector<bfloat16, 64> z_v_converted = z_v.to_vector<bfloat16>();
    ::aie::store_v(z, z_v_converted);
    z += 64;
  }
  event1();
}

void saxpy_scalar(bfloat16 *x, bfloat16 *y, bfloat16 *z) {
  event0();
  float a = 3.f;
  for (int i = 0; i < 4096; ++i) {
    z[i] = a * x[i] + y[i];
  }
  event1();
}
}

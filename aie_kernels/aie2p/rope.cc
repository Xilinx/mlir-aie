//===- rope.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, int N>
void rope_kernel(const T *restrict input, const T *restrict lut,
                 T *restrict output, int32_t dims) {
  event0();

  for (int v = 0; v < dims; v += N) {
    ::aie::vector<T, N> x = ::aie::load_v<N>(input + v);
    ::aie::vector<T, N> cache = ::aie::load_v<N>(lut + v);

    // Extract even and odd elements
    ::aie::vector<T, N / 2> x_even = ::aie::filter_even(x, 1);
    ::aie::vector<T, N / 2> x_odd = ::aie::filter_odd(x, 1);
    ::aie::vector<T, N / 2> cos_val = ::aie::filter_even(cache, 1);
    ::aie::vector<T, N / 2> sin_val = ::aie::filter_odd(cache, 1);

    // Perform ROPE calculations
    ::aie::vector<T, N / 2> even_cos = ::aie::mul(x_even, cos_val);
    ::aie::vector<T, N / 2> even_sin = ::aie::mul(x_even, sin_val);
    ::aie::vector<T, N / 2> odd_cos = ::aie::mul(x_odd, cos_val);
    ::aie::vector<T, N / 2> odd_sin = ::aie::mul(x_odd, sin_val);

    ::aie::vector<T, N / 2> output_even = ::aie::sub(even_cos, odd_sin);
    ::aie::vector<T, N / 2> output_odd = ::aie::add(even_sin, odd_cos);

    auto [low, high] = ::aie::interleave_zip(output_even, output_odd, 1);
    ::aie::vector<T, N> y = ::aie::concat(low, high);
    ::aie::store_v(output + v, y);
  }
  event1();
}

extern "C" {
void rope(bfloat16 *input, bfloat16 *lut, bfloat16 *output, int32_t dims) {
  rope_kernel<bfloat16, 16>(input, lut, output, dims);
}
}
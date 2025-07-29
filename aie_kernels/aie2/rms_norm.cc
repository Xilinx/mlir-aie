//===- rmsnorm.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, int N>
void rms_norm(const T *restrict input, T *restrict output, int32_t rows,
              int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;
  ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
  for (int r = 0; r < rows; r++) {
    T final_sum_sq = 0.0f;
    ::aie::vector<T, N> add_res = ::aie::broadcast<T, N>(0);
    for (int i = 0; i < cols; i = i + N) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + r * cols + i);
      ::aie::vector<T, N> square_v = ::aie::mul(reg_a, reg_a);
      add_res = ::aie::add(square_v, add_res);
    }
    for (int i = 0; i < N; i++)
      final_sum_sq = final_sum_sq + add_res[i];
    T inv_rms = aie::invsqrt(final_sum_sq / cols + epsilon);
    ::aie::vector<T, N> inv_rms_v = ::aie::broadcast<T, N>(inv_rms);

    for (int i = 0; i < cols; i = i + N) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + r * cols + i);
      ::aie::vector<T, N> norm_v = ::aie::mul(reg_a, inv_rms_v);
      ::aie::vector<T, N> out_v = ::aie::mul(norm_v, gamma_v);
      ::aie::store_v(output + r * cols + i, out_v);
    }
  }
  event1();
}

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t rows, int32_t cols) {
  rms_norm<bfloat16, 16>(input, output, rows, cols);
}
}
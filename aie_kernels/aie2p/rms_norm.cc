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

  for (int c = 0; c < cols; c++) {
    float sum_sq = 0.0f;
    for (int r = 0; r < rows; r++) {
      float x = float(input[r * cols + c]);
      sum_sq += x * x;
    }
    float inv_rms = aie::invsqrt(sum_sq / rows + epsilon);

    // Normalize each value in the column
    for (int r = 0; r < rows; r += N) {
      T *output_ptr = output + r * cols + c;
      const T *input_ptr = input + r * cols + c;
      ::aie::vector<T, N> input_v;
      for (int i = 0; i < N; i++) {
        input_v[i] = input_ptr[i * cols];
      }
      ::aie::vector<T, N> inv_rms_v = ::aie::broadcast<T, N>(inv_rms);
      ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
      ::aie::vector<T, N> norm_v = ::aie::mul(input_v, inv_rms_v);
      ::aie::vector<T, N> out_v = ::aie::mul(norm_v, gamma_v);
      for (int i = 0; i < N; i++) {
        output_ptr[i * cols] = out_v[i];
      }
    }
  }
  event1();
}

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t rows, int32_t cols) {
  rms_norm<bfloat16, 16>(input, output, rows, cols);
}
}

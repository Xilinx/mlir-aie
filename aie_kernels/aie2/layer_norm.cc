//===- layernorm.cc -------------------------------------------*- C++ -*-===//
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
void layer_norm(const T *restrict input, T *restrict output, int32_t rows,
                int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;
  const float beta = 0.0f;

  for (int c = 0; c < cols; c++) {
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;

    // Welford's pair-wise normalization algorithm for mean and variance
    for (int r = 0; r < rows; r++) {
      float x = float(input[r * cols + c]);
      count++;
      float delta = x - mean;
      mean += delta / count;
      float delta2 = x - mean;
      m2 += delta * delta2;
    }
    float variance = (count > 1) ? m2 / count : 0.0f;
    float inv_std = aie::invsqrt(variance + epsilon);

    // Normalize
    for (int r = 0; r < rows; r += N) {
      T *output_ptr = output + r * cols + c;
      const T *input_ptr = input + r * cols + c;
      ::aie::vector<T, N> input_v;
      for (int i = 0; i < N; i++) {
        input_v[i] = input_ptr[i * cols];
      }
      ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
      ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);
      ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
      ::aie::vector<T, N> beta_v = ::aie::broadcast<T, N>(beta);
      ::aie::vector<T, N> diff_v = ::aie::sub(input_v, mean_v);
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<T, N> scaled_v = ::aie::mul(norm_v, gamma_v);
      ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
      for (int i = 0; i < N; i++) {
        output_ptr[i * cols] = out_v[i];
      }
    }
  }
  event1();
}

extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t rows, int32_t cols) {
  layer_norm<bfloat16, 16>(input, output, rows, cols);
}
}
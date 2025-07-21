//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

inline float custom_sqrtf(float x) {
  if (x <= 0.0f)
    return 0.0f;
  float guess = x;
  // 5 iterations are enough for a reasonable precision
  for (int i = 0; i < 5; i++) {
    guess = 0.5f * (guess + x / guess);
  }
  return guess;
}

template <typename T, int N>
void layer_norm(const T *restrict input, T *restrict output, int32_t rows,
                int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f; // built-in constant
  const float beta = 0.0f;  // built-in constant
  for (int c = 0; c < cols; c++) {
    float mean = 0.0f;
    float m2 = 0.0f;
    int n = 0;
    // Welford's algorithm for mean and variance
    for (int r = 0; r < rows; r++) {
      float x = float(input[r * cols + c]);
      n++;
      float delta = x - mean;
      mean += delta / n;
      float delta2 = x - mean;
      m2 += delta * delta2;
    }
    float variance = (n > 1) ? (m2 / n) : 0.0f;
    float inv_std = aie::invsqrt(variance + epsilon);
    // Vectorized normalization using AIE intrinsics
    using vec_t = aie::vector<float, N>;
    int r = 0;
    // Broadcast mean, inv_std, gamma, beta to vectors
    vec_t vmean = aie::broadcast<float, N>(mean);
    vec_t vinv_std = aie::broadcast<float, N>(inv_std);
    vec_t vgamma = aie::broadcast<float, N>(gamma);
    vec_t vbeta = aie::broadcast<float, N>(beta);

    for (; r + N <= rows; r += N) {
      // Load N elements from input
      vec_t vin;
      for (int i = 0; i < N; i++) {
      vin[i] = float(input[(r + i) * cols + c]);
      }
      vec_t vnorm = aie::mul(aie::sub(vin, vmean), vinv_std);
      vec_t vout = aie::add(aie::mul(vnorm, vgamma), vbeta);
      for (int i = 0; i < N; i++) {
      output[(r + i) * cols + c] = T(vout[i]);
      }
    }
    // Handle remaining rows (if any)
    for (; r < rows; r++) {
      float x = float(input[r * cols + c]);
      float norm = (x - mean) * inv_std;
      float out = norm * gamma + beta;
      output[r * cols + c] = T(out);
    }
  }
  event1();
}
extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t rows, int32_t cols) {
  layer_norm<bfloat16, 16>(input, output, rows, cols);
}
}
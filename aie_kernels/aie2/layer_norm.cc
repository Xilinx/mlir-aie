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
void layer_norm(const T *restrict input, T *restrict output) {
  event0();
  constexpr int rows = 16;
  constexpr int cols = 64;
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f; // built-in constant
  const float beta = 0.0f;  // built-in constant
  for (int c = 0; c < cols; c++) {
    ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
    ::aie::accum<accfloat, N> sum_sq_acc = ::aie::zeros<accfloat, N>();
#pragma clang loop min_iteration_count(4)
    for (int r = 0; r < rows; r += N) {
      const T *input_ptr = input + r * cols + c;
      ::aie::vector<T, N> input_v;
      for (int i = 0; i < N; i++) {
        input_v[i] = input_ptr[i * cols];
      }
      ::aie::accum<accfloat, N> input_acc =
          ::aie::mul(input_v, ::aie::broadcast<T, N>(1.0f));
      sum_acc = ::aie::add(sum_acc, input_acc);
      ::aie::accum<accfloat, N> sq_acc = ::aie::mul(input_v, input_v);
      sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
    }
    // Convert accumulators to float arrays
    float sum_array[N];
    float sum_sq_array[N];
    for (int i = 0; i < N; i++) {
      sum_array[i] = float(sum_acc.template to_vector<float>()[i]);
      sum_sq_array[i] = float(sum_sq_acc.template to_vector<float>()[i]);
    }
    float sum_scalar = 0.0f;
    float sum_sq_scalar = 0.0f;
    for (int i = 0; i < N; i++) {
      sum_scalar += sum_array[i];
      sum_sq_scalar += sum_sq_array[i];
    }
    float mean = sum_scalar / float(rows);
    float variance = sum_sq_scalar / float(rows) - mean * mean;
    float inv_std = 1.0f / custom_sqrtf(variance + epsilon);
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
void layer_norm(bfloat16 *input, bfloat16 *output) {
  layer_norm<bfloat16, 16>(input, output);
}
}
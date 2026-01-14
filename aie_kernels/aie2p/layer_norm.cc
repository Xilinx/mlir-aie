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
void layer_norm(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;
  const float beta = 0.0f;

  ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
  ::aie::vector<T, N> beta_v = ::aie::broadcast<T, N>(beta);
  ::aie::vector<T, N> sum_acc = ::aie::zeros<T, N>();
  ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();

  int vector_chunks = cols / N;
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    sum_acc = ::aie::add(sum_acc, reg_a);
    ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
    sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
  }

  float sum_of_vals = ::aie::reduce_add(sum_acc);
  float sum_of_sq_vals = ::aie::reduce_add(sum_sq_acc);

  float mean = sum_of_vals / float(cols);
  float mean_sq = mean * mean;
  float variance = (sum_of_sq_vals / float(cols)) - mean_sq;
  float inv_std = aie::invsqrt(variance + epsilon);

  ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);
  ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
    ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
    ::aie::vector<T, N> scaled_v = ::aie::mul(norm_v, gamma_v);
    ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
    ::aie::store_v(output + i * N, out_v);
  }
  event1();
}

template <typename T, int N>
void layer_norm_welford(const T *restrict input, T *restrict output,
                        int32_t rows, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;
  const float beta = 0.0f;
  ::aie::vector<T, N> reg_a, delta, delta2, mean_v, m2_v, variance_v, just_div,
      just_prod;

  ::aie::vector<T, N> beta_v = ::aie::broadcast<T, N>(beta);
  ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
  ::aie::vector<T, N> epsilon_v = ::aie::broadcast<T, N>(epsilon);
  // Welford's algorithm for mean and variance, vectorized over columns
  float inv_count = 0.0f;
  mean_v = ::aie::zeros<T, N>();
  m2_v = ::aie::zeros<T, N>();
  for (int c = 0; c < cols; c += N) {
    for (int r = 0; r < rows; r++) {
      reg_a = ::aie::load_v<N>(input + r * cols + c);
      delta = ::aie::sub(reg_a, mean_v);
      inv_count = 1.0f / (r + 1);
      just_div = ::aie::mul(delta, inv_count);
      mean_v = ::aie::add(mean_v, just_div);
      delta2 = ::aie::sub(reg_a, mean_v);
      just_prod = ::aie::mul(delta, delta2);
      m2_v = ::aie::add(m2_v, just_prod);
    }
  }

  variance_v = ::aie::mul(m2_v, inv_count);
  ::aie::vector<T, N> var_eps_v = ::aie::add(variance_v, epsilon_v);
  ::aie::vector<T, N> inv_std_v = ::aie::invsqrt(var_eps_v);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c += N) {
      ::aie::vector<T, N> v0 = ::aie::load_v<N>(input + r * cols + c);
      ::aie::vector<T, N> diff_v = ::aie::sub(v0, mean_v);
      ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<T, N> scaled_v = ::aie::mul(norm_v, gamma_v);
      ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
      ::aie::store_v(output + r * cols + c, out_v);
    }
  }
  event1();
}

extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  layer_norm<bfloat16, 16>(input, output, cols);
}

void layer_norm_welford(float *input, float *output, int32_t rows,
                        int32_t cols) {
  layer_norm_welford<float, 16>(input, output, rows, cols);
}
}

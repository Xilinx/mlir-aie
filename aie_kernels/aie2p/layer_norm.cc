//===- layernorm.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

  int vector_chunks = cols / N;

  // Pass 1: mean = sum(x) / cols. Accumulate the running sum in an f32
  // accumulator, not a bf16 vector: a bf16 sum drops low-order bits as the
  // reduction length grows (embedding_dim is typically thousands), so the mean
  // itself is already lossy before the variance is even computed.
  ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    sum_acc = ::aie::add(sum_acc, reg_a);
  }
  float mean =
      ::aie::reduce_add(sum_acc.template to_vector<float>()) / float(cols);
  ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>((T)mean);

  // Pass 2: variance = sum((x - mean)^2) / cols. Centre first, then square
  // (the numerically stable two-pass form) instead of E[x^2] - mean^2, which
  // catastrophically cancels when mean^2 is close to E[x^2] -- the common case
  // for zero-ish-mean activations, and the reason this op needed a loose 0.1
  // tolerance.
  ::aie::accum<accfloat, N> var_acc = ::aie::zeros<accfloat, N>();
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
    ::aie::vector<float, N> sq = ::aie::mul_square(diff_v);
    var_acc = ::aie::add(var_acc, sq);
  }
  float variance =
      ::aie::reduce_add(var_acc.template to_vector<float>()) / float(cols);
  float inv_std = aie::invsqrt(variance + epsilon);

  ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>((T)inv_std);

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

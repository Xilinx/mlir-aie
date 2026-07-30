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

  // Reduce the row sum in an f32 accumulator, not a bf16 vector: a bf16 running
  // sum drops low-order bits as the reduction length grows (embedding_dim is
  // typically thousands), so the mean -- and every quantity derived from it --
  // is already lossy before the variance is computed. The sum of squares is
  // already reduced in f32.
  ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
  ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    sum_acc = ::aie::add(sum_acc, reg_a);
    ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
    sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
  }

  float mean =
      ::aie::reduce_add(sum_acc.template to_vector<float>()) / float(cols);
  float variance = ::aie::reduce_add(sum_sq_acc) / float(cols) - mean * mean;
  float inv_std = aie::invsqrt(variance + epsilon);

  ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>((T)mean);
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

// f32 per-row LayerNorm. The bf16 layer_norm above centers with a single
// E[x^2] - mean^2 reduction, which is fine because the bf16 input contract
// keeps the mean/std ratio small (a bf16 value near a large mean has an ulp
// wider than the std, so that regime is unrepresentable). On f32 input the mean
// can be large relative to the std and E[x^2] - mean^2 then catastrophically
// cancels, so this variant reduces over the feature axis (cols) per row with a
// numerically stable two-pass centered variance: center first, then square.
template <typename T, int N>
void layer_norm_f32(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;
  const float beta = 0.0f;

  ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
  ::aie::vector<T, N> beta_v = ::aie::broadcast<T, N>(beta);

  int vector_chunks = cols / N;

  // Pass 1: mean = sum(x) / cols.
  ::aie::vector<T, N> sum_v = ::aie::zeros<T, N>();
  for (int i = 0; i < vector_chunks; i++) {
    sum_v = ::aie::add(sum_v, ::aie::load_v<N>(input + i * N));
  }
  float mean = ::aie::reduce_add(sum_v) / float(cols);
  ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>(mean);

  // Pass 2: variance = sum((x - mean)^2) / cols (centered two-pass).
  ::aie::vector<T, N> var_v = ::aie::zeros<T, N>();
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> diff_v =
        ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<T, N> sq = ::aie::mul(diff_v, diff_v);
    var_v = ::aie::add(var_v, sq);
  }
  float variance = ::aie::reduce_add(var_v) / float(cols);
  float inv_std = aie::invsqrt(variance + epsilon);
  ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>(inv_std);

  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> diff_v =
        ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
    ::aie::vector<T, N> scaled_v = ::aie::mul(norm_v, gamma_v);
    ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
    ::aie::store_v(output + i * N, out_v);
  }
  event1();
}

extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  layer_norm<bfloat16, 16>(input, output, cols);
}

void layer_norm_f32(float *input, float *output, int32_t cols) {
  layer_norm_f32<float, 16>(input, output, cols);
}
}

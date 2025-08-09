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
void rms_norm(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;
  ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
  ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();
  ::aie::accum<acc32, N> acc = ::aie::zeros<acc32, N>();

  // Process data in vector chunks
  int vector_chunks = cols / N;
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
    acc = ::aie::add(add_res, square_v);
    add_res = acc.template to_vector<float>();
  }
  float sum_sq = ::aie::reduce_add(add_res);

  // Handle remaining elements
  int remaining = cols % N;
  if (remaining > 0) {
    int start_idx = vector_chunks * N;
    for (int i = 0; i < remaining; i++) {
      T val = input[start_idx + i];
      float square = static_cast<float>(val) * static_cast<float>(val);
      sum_sq += square;
    }
  }

  float rms = sum_sq / cols + epsilon;
  float inv_rms = aie::invsqrt(rms);
  ::aie::vector<T, N> inv_rms_v =
      ::aie::broadcast<T, N>(static_cast<T>(inv_rms));

  // Process vector chunks
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::vector<T, N> norm_v = ::aie::mul(reg_a, inv_rms_v);
    ::aie::vector<T, N> out_v = ::aie::mul(norm_v, gamma_v);
    ::aie::store_v(output + i * N, out_v);
  }

  // Handle remaining elements
  if (remaining > 0) {
    int start_idx = vector_chunks * N;
    for (int i = 0; i < remaining; i++) {
      T val = input[start_idx + i];
      T norm_val = static_cast<T>(static_cast<float>(val) * inv_rms);
      T out_val = static_cast<T>(static_cast<float>(norm_val) * gamma);
      output[start_idx + i] = out_val;
    }
  }
  event1();
}

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  rms_norm<bfloat16, 16>(input, output, cols);
}
}

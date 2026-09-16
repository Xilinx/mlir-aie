//===- rmsnorm.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vec_math.h>

template <typename T, int N>
void rms_norm(const T *restrict input, T *restrict output, int32_t cols,
              float epsilon = 1e-5f) {
  event0();
  ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();

  int vector_chunks = cols / N;
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
    add_res = ::aie::add(add_res, square_v);
  }
  float sum_sq = ::aie::reduce_add(add_res);

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
  float inv_rms = invsqrt(rms);
  // Peano has no f32 vector multiply for AIE2, so the f32 scale rides in a bf16
  // pair applied as two exact products accumulated in f32. A single bf16 scale
  // would shift every element of a norm the same way.
  T inv_rms_hi = static_cast<T>(inv_rms);
  T inv_rms_lo = static_cast<T>(inv_rms - static_cast<float>(inv_rms_hi));

  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::accum<accfloat, N> acc = ::aie::mul(reg_a, inv_rms_hi);
    acc = ::aie::mac(acc, reg_a, inv_rms_lo);
    ::aie::store_v(output + i * N, acc.template to_vector<T>());
  }

  if (remaining > 0) {
    int start_idx = vector_chunks * N;
    for (int i = 0; i < remaining; i++) {
      T val = input[start_idx + i];
      output[start_idx + i] = static_cast<T>(static_cast<float>(val) * inv_rms);
    }
  }
  event1();
}

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  rms_norm<bfloat16, 32>(input, output, cols);
}

void rms_norm_eps(bfloat16 *input, bfloat16 *output, int32_t cols,
                  float epsilon) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  rms_norm<bfloat16, 32>(input, output, cols, epsilon);
}
}

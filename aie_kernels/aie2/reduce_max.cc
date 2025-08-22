//===- reduce_max.cc --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <typename T, typename V>
void _reduce_max_vector(T *restrict in, T *restrict out,
                        const int32_t input_size) {
  event0();
  int32_t VECTOR_SIZE = V::size();
  V tiny = aie::broadcast<T>(std::numeric_limits<T>::lowest());
  V after_vector;
  V running_max = tiny;

  assert(input_size / VECTOR_SIZE >= 8);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t i = 0; i < input_size; i += VECTOR_SIZE) {
    V next = aie::load_v(in + i);
    V test = max(running_max, next);
    running_max = test;
  }

  after_vector = running_max;
  V first = shift_bytes(after_vector, after_vector, 32U);
  V second = max(after_vector, first);
  V second_shift = shift_bytes(second, second, 16U);
  V third = max(second, second_shift);
  V third_shift = shift_bytes(third, third, 8U);
  V fourth = max(third, third_shift);
  V fourth_shift = shift_bytes(fourth, fourth, 4U);
  V fifth = max(fourth, fourth_shift);
  if constexpr (std::is_same<V, aie::vector<bfloat16, 32>>::value) {
    V fifth_shift = shift_bytes(fifth, fifth, 2U);
    fifth = max(fifth, fifth_shift);
  }
  auto last = aie::reduce_max(fifth);
  *(T *)out = last;
  event1();
  return;
}
template <typename T>
void _reduce_max_scalar(T *restrict in, T *restrict out,
                        const int32_t input_size) {
  event0();
  T running_max = std::numeric_limits<T>::lowest();
  for (int32_t i = 0; i < input_size; i++) {
    if (in[i] > running_max)
      running_max = in[i];
  }
  *out = running_max;
  event1();

  return;
}

template <typename T>
void _compute_max(T *restrict in1, T *restrict in2, T *restrict out) {
  event0();
  *out = (*in1 > *in2) ? *in1 : *in2;
  event1();

  return;
}

extern "C" {

void reduce_max_vector_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                int32_t input_size) {
  _reduce_max_vector<bfloat16, aie::vector<bfloat16>>(a_in, c_out, input_size);
}

void reduce_max_scalar_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                int32_t input_size) {
  _reduce_max_scalar<bfloat16>(a_in, c_out, input_size);
}

void compute_max_bfloat16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  _compute_max<bfloat16>(a_in, b_in, c_out);
}

void reduce_max_vector(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_max_vector<int32_t, aie::vector<int32_t>>(a_in, c_out, input_size);
}

void reduce_max_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_max_scalar<int32_t>(a_in, c_out, input_size);
}

void compute_max(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  _compute_max<int32_t>(a_in, b_in, c_out);
}

} // extern "C"
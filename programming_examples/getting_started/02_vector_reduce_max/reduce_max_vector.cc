//===- reduce_max.cc --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T, typename V>
void _reduce_max_vector(T *restrict in, T *restrict out,
                        const int32_t input_size) {
  event0();
  int32_t VECTOR_SIZE = V::size();
  V tiny = aie::broadcast<T>(std::numeric_limits<T>::lowest());
  V after_vector;
  V running_max = tiny;

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

extern "C" {

void reduce_max_vector_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                int32_t input_size) {
  _reduce_max_vector<bfloat16, aie::vector<bfloat16>>(a_in, c_out, input_size);
}
} // extern "C"

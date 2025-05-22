//===- reduce_add.cc --------------------------------------------*- C++ -*-===//
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

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int N>
void reduce_add_scalar_int32(T_in *a, T_out *c) {
  event0();
  int32_t sum=0;
  for (int i = 0; i < N; i++) {
    sum = sum + a[i];
  }
  *(int32_t *)c = sum;
  event1();
}

template <typename T_in, typename T_out, const int N>
void reduce_add_vector_int32(T_in *in, T_out *out) {
  event0();
  v16int32 zero = broadcast_to_v16int32((int32_t)0);
  const int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_total = zero;
  for (int32_t i = 0; i < N; i += vector_size)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      v16int32 next = *(v16int32 *)(in + i);
      v16int32 test = add(running_total, next);
      running_total = test;
    }
  after_vector = running_total;
  v16int32 first = shift_bytes(after_vector, after_vector, 32U);
  v16int32 second = add(after_vector, first);
  v16int32 second_shift = shift_bytes(second, second, 16U);
  v16int32 third = add(second, second_shift);
  v16int32 third_shift = shift_bytes(third, third, 8U);
  v16int32 fourth = add(third, third_shift);
  v16int32 fourth_shift = shift_bytes(fourth, fourth, 4U);
  v16int32 fifth = add(fourth, fourth_shift);
  int32_t last = extract_elem(fifth, 0U);
  *(int32_t *)out = last;
  event1();
}

extern "C" {
void reduce_add_vector(int32_t *a_in, int32_t *c_out) {
  reduce_add_vector_int32<int32_t, int32_t, 256>(a_in, c_out);
}
void reduce_add_scalar(int32_t *a_in, int32_t *c_out) {
  reduce_add_scalar_int32<int32_t, int32_t, 256>(a_in, c_out);
}
} // extern "C"

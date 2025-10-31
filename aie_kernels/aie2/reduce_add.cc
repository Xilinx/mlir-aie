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

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

static void _reduce_add_scalar(int32_t *restrict in, int32_t *restrict out,
                               const int32_t input_size) {
  event0();
  int32_t running_total = 0;
  for (int32_t i = 0; i < input_size; i++) {
    running_total = running_total + in[i];
  }
  *out = running_total;
  event1();
  return;
}

static void _reduce_add_vector(int32_t *restrict in, int32_t *restrict out,
                               const int32_t input_size) {
  event0();
  v16int32 zero = broadcast_to_v16int32((int32_t)0);
  const int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_total = zero;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t i = 0; i < input_size; i += vector_size) {
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
  return;
}

extern "C" {
void reduce_add_vector(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_add_vector(a_in, c_out, input_size);
}
void reduce_add_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_add_scalar(a_in, c_out, input_size);
}
} // extern "C"

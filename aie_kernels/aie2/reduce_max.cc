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

void _reduce_max_vector(int32_t *restrict in, int32_t *restrict out,
                        const int32_t input_size) {

  event0();
  v16int32 tiny = broadcast_to_v16int32((int32_t)INT32_MIN);
  const int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_max = tiny;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t i = 0; i < input_size; i += vector_size) {
    v16int32 next = *(v16int32 *)(in + i);
    v16int32 test = max(running_max, next);
    running_max = test;
  }
  after_vector = running_max;
  v16int32 first = shift_bytes(after_vector, after_vector, 32U);
  v16int32 second = max(after_vector, first);
  v16int32 second_shift = shift_bytes(second, second, 16U);
  v16int32 third = max(second, second_shift);
  v16int32 third_shift = shift_bytes(third, third, 8U);
  v16int32 fourth = max(third, third_shift);
  v16int32 fourth_shift = shift_bytes(fourth, fourth, 4U);
  v16int32 fifth = max(fourth, fourth_shift);
  int32_t last = extract_elem(fifth, 0U);
  *(int32_t *)out = last;
  event1();
  return;
}

void _reduce_max_scalar(int32_t *restrict in, int32_t *restrict out,
                        const int32_t input_size) {
  event0();
  int32_t running_max = (int32_t)INT32_MIN;
  for (int32_t i = 0; i < input_size; i++) {
    if (in[i] > running_max)
      running_max = in[i];
  }
  *(int32_t *)out = running_max;
  event1();

  return;
}

void _compute_max(int32_t *restrict in1, int32_t *restrict in2,
                  int32_t *restrict out) {
  event0();
  if (*in1 > *in2)
    *out = *in1;
  else
    *out = *in2;
  event1();

  return;
}

extern "C" {

void reduce_max_vector(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_max_vector(a_in, c_out, input_size);
}

void reduce_max_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_max_scalar(a_in, c_out, input_size);
}

void compute_max(int32_t *a_in, int32_t *b_in, int32_t *c_out) {
  _compute_max(a_in, b_in, c_out);
}

} // extern "C"

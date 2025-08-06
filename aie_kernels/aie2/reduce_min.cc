//===- reduce_min.cc --------------------------------------------*- C++ -*-===//
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

void reduce_min_vector(int32_t *restrict in, int32_t *restrict out,
                       const int32_t input_size) {

  event0();
  v16int32 massive = broadcast_to_v16int32((int32_t)INT32_MAX);
  const int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_min = massive;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int32_t i = 0; i < input_size; i += vector_size) {
    v16int32 next = *(v16int32 *)(in + i);
    v16int32 test = min(running_min, next);
    running_min = test;
  }
  after_vector = running_min;
  v16int32 first = shift_bytes(after_vector, after_vector, 32U);
  v16int32 second = min(after_vector, first);
  v16int32 second_shift = shift_bytes(second, second, 16U);
  v16int32 third = min(second, second_shift);
  v16int32 third_shift = shift_bytes(third, third, 8U);
  v16int32 fourth = min(third, third_shift);
  v16int32 fourth_shift = shift_bytes(fourth, fourth, 4U);
  v16int32 fifth = min(fourth, fourth_shift);
  int32_t last = extract_elem(fifth, 0U);
  *(int32_t *)out = last;
  event1();
  return;
}

void reduce_min_scalar(int32_t *restrict in, int32_t *restrict out,
                       const int32_t input_size) {
  event0();
  int32_t running_min = (int32_t)INT32_MAX;
  for (int32_t i = 0; i < input_size; i++) {
    if (in[i] < running_min)
      running_min = in[i];
  }
  *(int32_t *)out = running_min;
  event1();

  return;
}

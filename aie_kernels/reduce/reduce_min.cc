//===- reduce_min.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef REDUCE_MIN_ELEMS
#define REDUCE_MIN_ELEMS input_size
#endif

void _reduce_min_vector(int32_t *restrict in, int32_t *restrict out,
                        const int32_t input_size) {

  event0();
  v16int32 massive = broadcast_to_v16int32((int32_t)INT32_MAX);
  const int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_min = massive;
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  // Walked pointer, rolled loop: see reduce_add.cc.
  const v16int32 *p = (const v16int32 *)in;
  AIE_LOOP_NO_UNROLL
  for (int32_t i = 0; i < REDUCE_MIN_ELEMS; i += vector_size)
    running_min = min(running_min, *p++);
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int32_t i = 0; i < REDUCE_MIN_ELEMS; i += vector_size) {
    v16int32 next = *(v16int32 *)(in + i);
    v16int32 test = min(running_min, next);
    running_min = test;
  }
#endif
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

void _reduce_min_scalar(int32_t *restrict in, int32_t *restrict out,
                        const int32_t input_size) {
  event0();
  int32_t running_min = (int32_t)INT32_MAX;
  for (int32_t i = 0; i < REDUCE_MIN_ELEMS; i++) {
    if (in[i] < running_min)
      running_min = in[i];
  }
  *(int32_t *)out = running_min;
  event1();

  return;
}

// bf16 min, as reduce_max.cc's bf16 max.
static void _reduce_min_vector_bf16(bfloat16 *restrict in,
                                    bfloat16 *restrict out,
                                    const int32_t input_size) {
  event0();
  constexpr int32_t vector_size = 32;
  aie::vector<bfloat16, vector_size> running_min =
      aie::broadcast<bfloat16, vector_size>(
          std::numeric_limits<bfloat16>::max());
  const bfloat16 *p = in;
  AIE_LOOP_NO_UNROLL
  for (int32_t i = 0; i < REDUCE_MIN_ELEMS; i += vector_size) {
    running_min = aie::min(running_min, aie::load_v<vector_size>(p));
    p += vector_size;
  }
  *out = aie::reduce_min(running_min);
  event1();
}

static void _reduce_min_scalar_bf16(bfloat16 *restrict in,
                                    bfloat16 *restrict out,
                                    const int32_t input_size) {
  event0();
  bfloat16 running_min = std::numeric_limits<bfloat16>::max();
  for (int32_t i = 0; i < REDUCE_MIN_ELEMS; i++) {
    if (in[i] < running_min)
      running_min = in[i];
  }
  *out = running_min;
  event1();
}

extern "C" {

void reduce_min_vector_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                int32_t input_size) {
  _reduce_min_vector_bf16(a_in, c_out, input_size);
}

void reduce_min_scalar_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                int32_t input_size) {
  _reduce_min_scalar_bf16(a_in, c_out, input_size);
}

void reduce_min_vector(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_min_vector(a_in, c_out, input_size);
}

void reduce_min_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_min_scalar(a_in, c_out, input_size);
}

} // extern "C"

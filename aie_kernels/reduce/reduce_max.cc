//===- reduce_max.cc --------------------------------------------*- C++ -*-===//
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

#ifndef REDUCE_MAX_ELEMS
#define REDUCE_MAX_ELEMS input_size
#endif

template <typename T, typename V>
void _reduce_max_vector(T *restrict in, T *restrict out,
                        const int32_t input_size) {
  event0();
  constexpr int32_t VECTOR_SIZE = V::size();
  V tiny = aie::broadcast<T>(std::numeric_limits<T>::lowest());
  V running_max = tiny;

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  // Walked pointer, rolled loop: see reduce_add.cc.
  const T *p = in;
  AIE_LOOP_NO_UNROLL
  for (int32_t i = 0; i < REDUCE_MAX_ELEMS; i += VECTOR_SIZE) {
    running_max = max(running_max, aie::load_v<VECTOR_SIZE>(p));
    p += VECTOR_SIZE;
  }
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int32_t i = 0; i < REDUCE_MAX_ELEMS; i += VECTOR_SIZE) {
    V next = aie::load_v(in + i);
    V test = max(running_max, next);
    running_max = test;
  }
#endif

#if AIE_TUNED_AIE2P
  *(T *)out = aie::reduce_max(running_max);
#else
  V after_vector = running_max;
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
#endif
  event1();
  return;
}
template <typename T>
void _reduce_max_scalar(T *restrict in, T *restrict out,
                        const int32_t input_size) {
  event0();
  T running_max = std::numeric_limits<T>::lowest();
  for (int32_t i = 0; i < REDUCE_MAX_ELEMS; i++) {
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

//===- reduce_add.cc --------------------------------------------*- C++ -*-===//
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

#ifndef REDUCE_ADD_ELEMS
#define REDUCE_ADD_ELEMS input_size
#endif

static void _reduce_add_scalar(int32_t *restrict in, int32_t *restrict out,
                               const int32_t input_size) {
  event0();
  int32_t running_total = 0;
  for (int32_t i = 0; i < REDUCE_ADD_ELEMS; i++) {
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
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  // A walked pointer and a rolled loop pipeline; indexing `in + i` unrolls
  // into one serial chain.
  const v16int32 *p = (const v16int32 *)in;
  AIE_LOOP_NO_UNROLL
  for (int32_t i = 0; i < REDUCE_ADD_ELEMS; i += vector_size)
    running_total = add(running_total, *p++);
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int32_t i = 0; i < REDUCE_ADD_ELEMS; i += vector_size) {
    v16int32 next = *(v16int32 *)(in + i);
    v16int32 test = add(running_total, next);
    running_total = test;
  }
#endif
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

// A bf16 sum accumulates in fp32 and rounds to bf16 once, on the result, to
// nearest even (the core's default mode rounds down).
static bfloat16 _round_bf16(float x) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  bfloat16 r = (bfloat16)x;
  aie::set_rounding(saved);
  return r;
}

static void _reduce_add_vector_bf16(bfloat16 *restrict in,
                                    bfloat16 *restrict out,
                                    const int32_t input_size) {
  event0();
  constexpr int32_t L = AIE_BF16_LANES;
  const bfloat16 *p = in;
  aie::accum<accfloat, L> t0 = aie::zeros<accfloat, L>();
  if constexpr (REDUCE_ADD_ELEMS % (4 * L) == 0) {
    // The adds chain on their latency; four independent sums hide it.
    aie::accum<accfloat, L> t1 = t0, t2 = t0, t3 = t0;
    AIE_LOOP_NO_UNROLL
    for (int32_t i = 0; i < REDUCE_ADD_ELEMS; i += 4 * L) {
      t0 = aie::add(t0, aie::load_v<L>(p));
      t1 = aie::add(t1, aie::load_v<L>(p + L));
      t2 = aie::add(t2, aie::load_v<L>(p + 2 * L));
      t3 = aie::add(t3, aie::load_v<L>(p + 3 * L));
      p += 4 * L;
    }
    t0 = aie::add(t0, t1.to_vector<float>());
    t2 = aie::add(t2, t3.to_vector<float>());
    t0 = aie::add(t0, t2.to_vector<float>());
  } else {
    AIE_LOOP_NO_UNROLL
    for (int32_t i = 0; i < REDUCE_ADD_ELEMS; i += L) {
      t0 = aie::add(t0, aie::load_v<L>(p));
      p += L;
    }
  }
  *out = _round_bf16(aie::reduce_add(t0.to_vector<float>()));
  event1();
}

static void _reduce_add_scalar_bf16(bfloat16 *restrict in,
                                    bfloat16 *restrict out,
                                    const int32_t input_size) {
  event0();
  float running_total = 0.0f;
  for (int32_t i = 0; i < REDUCE_ADD_ELEMS; i++)
    running_total += (float)in[i];
  *out = _round_bf16(running_total);
  event1();
}

extern "C" {
void reduce_add_vector_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                int32_t input_size) {
  _reduce_add_vector_bf16(a_in, c_out, input_size);
}
void reduce_add_scalar_bfloat16(bfloat16 *a_in, bfloat16 *c_out,
                                int32_t input_size) {
  _reduce_add_scalar_bf16(a_in, c_out, input_size);
}
void reduce_add_vector(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_add_vector(a_in, c_out, input_size);
}
void reduce_add_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_add_scalar(a_in, c_out, input_size);
}
} // extern "C"

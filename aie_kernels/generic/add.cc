//===- add.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef ADD_ELEMS
#define ADD_ELEMS size
#endif

// One bf16 vector register: 512 bits on AIE2P, 256 on AIE2.
#if __AIE_ARCH__ >= 21
#define ADD_VEC_FACTOR 32
#else
#define ADD_VEC_FACTOR 16
#endif

template <typename T_in, typename T_out, const int N>
void eltwise_add(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

// Four independent load/add/store chains per iteration.
//
// One chain per iteration is latency-bound rather than issue-bound: the two
// operand loads are both vlda.conv.fp32.bf16, which only exists on the a port,
// so they serialize, and the vadd.f then waits out the load-to-use latency
// with nothing to fill it -- three of the loop's seven bundles were nops.
// Peano schedules the four-chain body into a loop of the same size, so the
// extra chains land in stall slots that were already being paid for.  Going
// wider is not free: x8 grows the body faster than the work it adds, and x16
// runs out of accumulator registers.
#define ADD_UNROLL 4

template <typename T_in, typename T_out, const int N>
void eltwise_vadd(T_in *a, T_in *b, T_out *c) {

  constexpr int vec_factor = ADD_VEC_FACTOR;
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  constexpr int F = N / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / ADD_UNROLL; i++) {
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = aie::add(A0, B0);
    *pC1++ = aie::add(A1, B1);
    *pC1++ = aie::add(A2, B2);
    *pC1++ = aie::add(A3, B3);
  }
  // Whole vectors past the last full group of four.  N is a compile-time
  // constant, so for the 1024-element tile the factories build this is zero
  // iterations and folds away entirely.
  for (int i = 0; i < F % ADD_UNROLL; i++) {
    *pC1++ = aie::add(*pA1++, *pB1++);
  }
  event1();
}

// Runtime size (need not divide vec_factor); scalar tail avoids the full-width
// load_v/store_v reading/writing past the buffer on a short final vector.
template <typename T_in, typename T_out>
void eltwise_vadd_size(T_in *a, T_in *b, T_out *c, int size) {
  constexpr int vec_factor = ADD_VEC_FACTOR;
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  const int F = ADD_ELEMS / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / ADD_UNROLL; i++) { // see eltwise_vadd
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = aie::add(A0, B0);
    *pC1++ = aie::add(A1, B1);
    *pC1++ = aie::add(A2, B2);
    *pC1++ = aie::add(A3, B3);
  }
  for (int i = 0; i < F % ADD_UNROLL; i++) {
    *pC1++ = aie::add(*pA1++, *pB1++);
  }
  // Scalar tail for a size that is not a whole number of vectors.  Index off
  // the base pointers rather than the iterators: the vector body consumed
  // exactly F vectors, so the leftover elements start at F * vec_factor.
  const int done = F * vec_factor;
  const int tail = ADD_ELEMS - done;
  for (int i = 0; i < tail; i++) {
    c[done + i] = a[done + i] + b[done + i];
  }
  event1();
}

extern "C" {

void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_add<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vadd<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_add_bf16_vector_size(bfloat16 *a_in, bfloat16 *b_in,
                                  bfloat16 *c_out, int size) {
  eltwise_vadd_size<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

} // extern "C"

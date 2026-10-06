//===- mul.cc -------------------------------------------------*- C++ -*-===//
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

#ifndef MUL_ELEMS
#define MUL_ELEMS size
#define MUL_ELEMS_RUNTIME
#endif

template <typename T_in, typename T_out, const int N>
void eltwise_mul(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}

// The untuned path runs four independent load/mul/store chains per iteration;
// see ADD_UNROLL in add.cc.
#define MUL_UNROLL 4

// aie::mul on bf16 yields an accumulator (fp32 products); convert back to
// T_out explicitly.  Assigning the accumulator straight into a vector<T_out>
// produces garbage at the 32-wide AIE2P width.
#define MUL_ONE(A, B) (aie::mul((A), (B)).template to_vector<T_out>())

// The tuned paths run one chain per iteration instead, which pipelines (II1
// on AIE2P) with restrict pointers and a rolled loop.
template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *__restrict a, T_in *__restrict b, T_out *__restrict c) {

  constexpr int vec_factor = AIE_BF16_LANES;
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  constexpr int F = N / vec_factor;
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < F / MUL_UNROLL * MUL_UNROLL; i++) {
    *pC1++ = MUL_ONE(*pA1++, *pB1++);
  }
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / MUL_UNROLL; i++) {
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = MUL_ONE(A0, B0);
    *pC1++ = MUL_ONE(A1, B1);
    *pC1++ = MUL_ONE(A2, B2);
    *pC1++ = MUL_ONE(A3, B3);
  }
#endif
  // Whole vectors past the last full group of four.
  for (int i = 0; i < F % MUL_UNROLL; i++) {
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    *pC1++ = MUL_ONE(A0, B0);
  }
  event1();
}

// Runtime size (need not divide vec_factor); scalar tail avoids the full-width
// load_v/store_v reading/writing past the buffer on a short final vector.
template <typename T_in, typename T_out>
void eltwise_vmul_size(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c, int size) {
  constexpr int vec_factor = AIE_BF16_LANES;
  event0();
  auto pA1 = aie::begin_restrict_vector<vec_factor>(a);
  auto pB1 = aie::begin_restrict_vector<vec_factor>(b);
  auto pC1 = aie::begin_restrict_vector<vec_factor>(c);
  const int F = (uint32_t)MUL_ELEMS / vec_factor; // see eltwise_vadd_size
// The single chain pipelines only with a compile-time trip count.
#if (AIE_TUNED_AIE2 || AIE_TUNED_AIE2P) && !defined(MUL_ELEMS_RUNTIME)
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < F / MUL_UNROLL * MUL_UNROLL; i++) {
    *pC1++ = MUL_ONE(*pA1++, *pB1++);
  }
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < F / MUL_UNROLL; i++) { // see eltwise_vmul
    auto A0 = *pA1++;
    auto B0 = *pB1++;
    auto A1 = *pA1++;
    auto B1 = *pB1++;
    auto A2 = *pA1++;
    auto B2 = *pB1++;
    auto A3 = *pA1++;
    auto B3 = *pB1++;
    *pC1++ = MUL_ONE(A0, B0);
    *pC1++ = MUL_ONE(A1, B1);
    *pC1++ = MUL_ONE(A2, B2);
    *pC1++ = MUL_ONE(A3, B3);
  }
#endif
  if ((uint32_t)MUL_ELEMS % (vec_factor * MUL_UNROLL)) {
    for (int i = 0; i < F % MUL_UNROLL; i++) {
      auto A0 = *pA1++;
      auto B0 = *pB1++;
      *pC1++ = MUL_ONE(A0, B0);
    }
    // Scalar tail for a size that is not a whole number of vectors.
    const int done = F * vec_factor;
    const int tail = MUL_ELEMS - done;
    for (int i = 0; i < tail; i++) {
      c[done + i] = a[done + i] * b[done + i];
    }
  }
  event1();
}

extern "C" {

void eltwise_mul_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_mul<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vmul<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_mul_bf16_vector_size(bfloat16 *a_in, bfloat16 *b_in,
                                  bfloat16 *c_out, int size) {
  eltwise_vmul_size<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

} // extern "C"

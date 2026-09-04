//===- mul.cc -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
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

template <typename T_in, typename T_out, const int N>
void eltwise_mul(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
  }
}

template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *a, T_in *b, T_out *c) {

  // 32 bf16 = 512 bits = one AIE2P vector register (AIE2's is 256-bit and uses
  // a 16-wide loop; see aie2/mul.cc).
  constexpr int vec_factor = 32;
  event0();
  T_in *__restrict pA1 = a;
  T_in *__restrict pB1 = b;
  T_out *__restrict pC1 = c;
  const int F = N / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
    pB1 += vec_factor;
    // aie::mul on bf16 yields an accumulator (fp32 products); convert back to
    // T_out explicitly.  Assigning the accumulator straight into a
    // vector<T_out> produces garbage at this 32-wide width.
    aie::vector<T_out, vec_factor> cout =
        aie::mul(A0, B0).template to_vector<T_out>();
    aie::store_v(pC1, cout);
    pC1 += vec_factor;
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

} // extern "C"

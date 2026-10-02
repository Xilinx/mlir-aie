//===- scale_shift.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int N>
void eltwise_mul_add(T_in *a, T_in *b, T_out *c, bool is_mul) {
  if (is_mul) {
    for (int i = 0; i < N; i++) {
      c[i] = a[i] * b[i];
    }
  } else {
    for (int i = 0; i < N; i++) {
      c[i] = a[i] + b[i];
    }
  }
}

// Tuned: restrict parameters and a rolled loop let the pipeliner overlap
// iterations; the add goes through a mac so only a needs the a-port-only
// vlda.conv (see add.cc).
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
template <typename T_in, typename T_out, const int N>
void eltwise_vadd(T_in *__restrict a, T_in *__restrict b, T_out *__restrict c) {
  constexpr int vec_factor = AIE_BF16_LANES;
  event0();
  auto pA = aie::begin_restrict_vector<vec_factor>(a);
  auto pB = aie::begin_restrict_vector<vec_factor>(b);
  auto pC = aie::begin_restrict_vector<vec_factor>(c);
  const auto ones = aie::broadcast<T_in, vec_factor>(1.0f);
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < N / vec_factor; i++) {
    aie::accum<accfloat, vec_factor> acc;
    acc.from_vector(*pA++);
    *pC++ = aie::mac(acc, *pB++, ones).template to_vector<T_out>();
  }
  event1();
}

template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *__restrict a, T_in *__restrict b, T_out *__restrict c) {
  constexpr int vec_factor = AIE_BF16_LANES;
  event0();
  auto pA = aie::begin_restrict_vector<vec_factor>(a);
  auto pB = aie::begin_restrict_vector<vec_factor>(b);
  auto pC = aie::begin_restrict_vector<vec_factor>(c);
  AIE_LOOP_NO_UNROLL
  for (int i = 0; i < N / vec_factor; i++) {
    *pC++ = aie::mul(*pA++, *pB++).template to_vector<T_out>();
  }
  event1();
}
#else
template <typename T_in, typename T_out, const int N>
void eltwise_vadd(T_in *a, T_in *b, T_out *c) {

  constexpr int vec_factor = 16;
  event0();
  T_in *__restrict pA1 = a;
  T_in *__restrict pB1 = b;
  T_out *__restrict pC1 = c;
  const int F = N / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
      pB1 += vec_factor;
      aie::vector<T_out, vec_factor> cout = aie::add(A0, B0);
      aie::store_v(pC1, cout);
      pC1 += vec_factor;
    }
  event1();
}
template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *a, T_in *b, T_out *c) {

  constexpr int vec_factor = 16;
  event0();
  T_in *__restrict pA1 = a;
  T_in *__restrict pB1 = b;
  T_out *__restrict pC1 = c;
  const int F = N / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
      pB1 += vec_factor;
      aie::vector<T_out, vec_factor> cout = aie::mul(A0, B0);
      aie::store_v(pC1, cout);
      pC1 += vec_factor;
    }
  event1();
}
#endif

template <typename T_in, typename T_out, const int N>
void eltwise_vmul_vadd(T_in *a, T_in *b, T_out *c, bool is_mul) {
  if (is_mul) {
    eltwise_vmul<T_in, T_out, N>(a, b, c);
  } else {
    eltwise_vadd<T_in, T_out, N>(a, b, c);
  }
}

extern "C" {

void eltwise_mul_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in,
                                 bfloat16 *c_out, int32 is_mul) {
  eltwise_mul_add<bfloat16, bfloat16, 1024>(a_in, b_in, c_out, bool(is_mul));
}

void eltwise_mul_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in,
                                 bfloat16 *c_out, int32 is_mul) {
  eltwise_vmul_vadd<bfloat16, bfloat16, 1024>(a_in, b_in, c_out, bool(is_mul));
}

} // extern "C"

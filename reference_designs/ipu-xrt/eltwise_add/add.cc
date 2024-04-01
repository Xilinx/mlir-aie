//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int N>
void eltwise_add(T_in *a, T_in *b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

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

extern "C" {

void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_add<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  eltwise_vadd<bfloat16, bfloat16, 1024>(a_in, b_in, c_out);
}

} // extern "C"

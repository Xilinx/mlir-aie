//===- scale.cc -------------------------------------------------*- C++ -*-===//
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

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

template <typename T_in, typename T_out, const int N>
void scale(T_in *a, T_out *c, T_in factor) {
  event0();
  for (int i = 0; i < N; i++) {
    c[i] = factor * a[i];
  }
  event1();
}

// Assume factor is at least 16
template <typename T_in, typename T_out, const int N>
void scale_vectorized(T_in *a, T_out *c, T_in factor) {
  constexpr int vec_factor = 16;
  event0();
  T_in *__restrict pA1 = a;
  T_out *__restrict pC1 = c;
  const int F = N / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::accum<acc64, vec_factor> cout = aie::mul(A0, factor);
      aie::store_v(pC1, cout.to_vector<T_out>(0));
      pC1 += vec_factor;
    }
  event1();
}

extern "C" {

void scale_int32(int32_t *a_in, int32_t *c_out) {
  scale_vectorized<int32_t, int32_t, 1024>(a_in, c_out, 3);
}

void scale_scalar_int32(int32_t *a_in, int32_t *c_out) {
  scale<int32_t, int32_t, 1024>(a_in, c_out, 3);
}

} // extern "C"

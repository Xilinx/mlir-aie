//===- sum.cc ----------------------------------------------*- C++
////-*-===//
////
//// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
//// See https://llvm.org/LICENSE.txt for license information.
//// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
//// Copyright (C) 2025, Advanced Micro Devices, Inc.
////
////===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T>
void zero_vectorized(T *__restrict c) {
  const aie::vector<T, 16> zeros = aie::zeros<T, 16>();
  aie::store_v(c, zeros);
}

void sum_kernel(int32_t *a, int32_t *c) {
  event0();

  const aie::vector<int32_t, 16> a_vec = aie::load_v<16>(a);
  const aie::vector<int32_t, 16> c_vec = aie::load_v<16>(c);
  aie::vector<int32_t, 16> sum_vec = aie::add(c_vec, a_vec);
  aie::store_v(c, sum_vec);

  event1();
}
extern "C" {

void sum(int32_t *a, int32_t *c) { sum_kernel(a, c); }

void zero(int32_t *restrict c) { zero_vectorized<int32_t>(c); }

} // extern "C"

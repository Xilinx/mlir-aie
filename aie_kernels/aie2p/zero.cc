//===- zero.cc --------------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZERO_CC
#define ZERO_CC

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T, int M, int N>
void zero_scalar(T *__restrict c) {
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8); // 512 bit store units for AIE2P
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  event0();
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
  event1();
}

#endif
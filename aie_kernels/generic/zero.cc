//===- zero.cc --------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_GENERIC_ZERO_CC
#define AIE_KERNELS_GENERIC_ZERO_CC

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

// Bits one store instruction writes: 512 on AIE2P, 256 on AIE2. Peano defines
// __AIEARCH__ as 21 and 20 respectively, so the width follows the target
// rather than the file, and one implementation serves both.
#if __AIEARCH__ >= 21
#define AIE_STORE_BITS 512
#else
#define AIE_STORE_BITS 256
#endif

template <typename T, int M, int N>
void zero_scalar(T *__restrict c) {
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = AIE_STORE_BITS / (sizeof(T) * 8);
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

//===- zero.cc --------------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_AIE2P_ZERO_CC
#define AIE_KERNELS_AIE2P_ZERO_CC

#include <aie_api/aie.hpp>
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

// Width is the store width in bits. 512 is the AIE2P native store unit and the
// right default; a narrower one lets a buffer whose element count is not a
// multiple of the 512-bit lane count still be zeroed vectorized.
template <typename T, int M, int N, int Width = 512>
void zero_vectorized(T *__restrict c) {
  constexpr int r = Width / (sizeof(T) * 8);
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

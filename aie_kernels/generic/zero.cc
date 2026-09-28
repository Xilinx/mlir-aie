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

template <typename T, int M, int N>
void zero_scalar(T *__restrict c) {
  // The unroll is disabled to work around a Peano miscompile, not for speed.
  // Unrolled by two, a 16-bit store loop emits a bundle that reads dj0 for the
  // store and writes dj0 in the same bundle; on hardware the second store then
  // lands on the first store's address, leaving every other element holding
  // the loop's own byte offset. Measured on aie2p for int16 at every M*N > 32
  // with M*N % 4 == 2 -- 34, 38, 62, 66, 70, 98 -- and correct everywhere else,
  // including 8- and 32-bit stores at the same counts. This is the scalar
  // fallback, so the lost unroll costs nothing a caller asked for.
#pragma clang loop unroll(disable)
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = aie::native_vector_length_v<T>;
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  event0();
  int i = 0;
  for (; i + r <= M * N; i += r) {
    aie::store_v(c + i, zeros);
  }
  for (; i < M * N; ++i) {
    c[i] = 0;
  }
  event1();
}

#if defined(ZERO_TYPE) && defined(TILE_SIZE)
extern "C" void zero(ZERO_TYPE *__restrict output) {
#ifdef ZERO_SCALAR
  zero_scalar<ZERO_TYPE, TILE_SIZE, 1>(output);
#else
  zero_vectorized<ZERO_TYPE, TILE_SIZE, 1>(output);
#endif
}
#endif

#endif

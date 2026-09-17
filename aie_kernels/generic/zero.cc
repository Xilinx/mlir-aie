//===- zero.cc --------------------------------------------000---*- C++ -*-===//
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

// The native store unit: 512-bit on AIE2P, 256-bit on AIE2. This was the only
// difference between the two per-arch copies this file replaces.
#if __AIE_ARCH__ >= 21
#define ZERO_STORE_BITS 512
#else
#define ZERO_STORE_BITS 256
#endif

template <typename T, int M, int N>
void zero_scalar(T *__restrict c) {
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

// Width is the store width in bits. The arch-native default is almost always
// right; a narrower one lets a buffer whose element count is not a multiple of
// the native lane count still be zeroed vectorized.
template <typename T, int M, int N, int Width = ZERO_STORE_BITS>
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

// C-linkage wrappers for the `combos()` table in the matmul kernels. These
// expand against the including translation unit's DIM_M / DIM_N, so they must
// be used where those are in scope. mv.cc has its own pair: its output is a
// vector, so the N is 1 and the symbol is spelled differently.
#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out) {                                \
    zero_vectorized<ctype_out, DIM_M, DIM_N>(c_out);                           \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t)                                            \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_M, DIM_N>(c_out);                               \
  }

#endif

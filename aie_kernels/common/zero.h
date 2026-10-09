//===- zero.h ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_ZERO_H
#define AIE_KERNELS_COMMON_ZERO_H

#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, int M, int N>
void zero_scalar(T *__restrict c) {
  // Unrolled, Peano reuses the incremented address and skips every other
  // 16-bit store (the int16 tile_size=34 case).
#pragma clang loop unroll(disable)
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

// Zeroes p[0, n) for n under one 128-bit register, from a 4-byte aligned p:
// whole 32-bit words, then elements. store_unaligned_v would be shorter, but
// it rewrites the 64 bytes around its address, past the end of p.
template <typename T>
inline void zero_sub_vector(T *__restrict p, int n) {
  if constexpr (sizeof(T) < 4) {
    using word = int32_t __attribute__((may_alias));
    constexpr int per_word = 4 / sizeof(T);
    word *__restrict w = (word *)p;
    for (int i = 0; i < n / per_word; ++i)
      w[i] = 0;
    p += n / per_word * per_word;
    n %= per_word;
  }
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; ++i)
    p[i] = 0;
}

// `markers = false` leaves the bracketing to a caller that times a larger call.
template <typename T, int M, int N, bool markers = true>
void zero_vectorized(T *__restrict c) {
  constexpr int r = aie::native_vector_length_v<T>;
  constexpr int q = 16 / sizeof(T);
  constexpr int n = M * N;
  // Unroll only small tiles, where the loop's bookkeeping outweighs the stores.
  constexpr int unroll = (n / r >= 1 && n / r <= 16) ? n / r : 1;
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  // A walking cursor, so the store can post-increment.
  T *__restrict p = c;
  if constexpr (markers)
    event0();
#pragma clang loop unroll_count(unroll)
  for (int i = 0; i < n / r; ++i, p += r) {
    aie::store_v(p, zeros);
  }
  for (int i = 0; i < n % r / q; ++i, p += q) {
    aie::store_v(p, aie::zeros<T, q>());
  }
  zero_sub_vector(p, n % q);
  if constexpr (markers)
    event1();
}

#endif

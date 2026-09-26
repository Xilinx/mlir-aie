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
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }
}

// Narrowest vector covering `rem` elements, never below one 128-bit register.
template <typename T>
constexpr int zero_tail_width(int rem) {
  int w = 16 / sizeof(T);
  while (w < rem)
    w *= 2;
  return w;
}

// `markers = false` leaves the bracketing to a caller that times a larger call.
template <typename T, int M, int N, bool markers = true>
void zero_vectorized(T *__restrict c) {
  constexpr int r = aie::native_vector_length_v<T>;
  constexpr int n = M * N;
  constexpr int w = zero_tail_width<T>(n % r);
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
  if constexpr (n % r != 0) {
    if constexpr (n >= w) {
      // Overlap elements the body already wrote rather than a scalar loop; a
      // zero stored twice is still zero.
      aie::store_unaligned_v(c + n - w, aie::zeros<T, w>());
    } else {
      for (int i = (n / r) * r; i < n; ++i) {
        c[i] = 0;
      }
    }
  }
  if constexpr (markers)
    event1();
}

#endif

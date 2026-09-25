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
  // A tile is only a handful of stores, and below this count the loop's own
  // bookkeeping outweighs them: unrolled, int32/TILE_SIZE=64 drops from 176 to
  // 48 bytes of .text. Past it the stores outweigh the loop, so callers with
  // big accumulators (mha, flash_attn_prefill, mm_fused) keep the rolled form.
  constexpr int unroll = (n / r >= 1 && n / r <= 16) ? n / r : 1;
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  // A walking cursor rather than `c + i * r`: it lets the store use the
  // post-increment addressing mode, so a rolled tile is one bundle per store
  // (II 1) instead of a `movs dj0` / `add` address chain the scheduler can only
  // overlap down to II 2.
  T *__restrict p = c;
  if constexpr (markers)
    event0();
#pragma clang loop unroll_count(unroll)
  for (int i = 0; i < n / r; ++i, p += r) {
    aie::store_v(p, zeros);
  }
  if constexpr (n % r != 0) {
    if constexpr (n >= w) {
      // Reach back over elements the body already wrote rather than closing
      // with a scalar loop: at TILE_SIZE=108 the 44-element remainder is
      // scheduled as its own II-30 loop, longer than the body it follows, and
      // a zero stored twice is still zero.
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

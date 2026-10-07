//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_arch.h"

// Each header's micro-tiles are its architecture's mmul shapes, and AIE2 has
// not all of AIE2P's (no transpose of an 8x8 int32 C tile), so the choice
// follows the architecture rather than AIE_TUNED_*. _MM_MAC_DIMS in
// python/iron/kernels/linalg.py is keyed the same way.
#if AIE_ARCH_AIE2
#include "mm_aie2.h"
#else
#include "mm_aie2p.h"
#endif

// Zeroes, in each of `runs` runs `run_stride` apart from `p`, `tiles`
// micro-tiles of `L` elements, the first `partial` of them only in the lanes
// `lo` and `hi` do not set. Out of line and for size: inlined, Peano unrolls
// it over the constant micro-tile counts at each call.
template <typename T, unsigned L>
__attribute__((noinline, minsize)) static void
zero_micro_tiles(T *__restrict p, int32_t runs, int32_t run_stride,
                 int32_t tiles, int32_t partial, uint32_t lo, uint32_t hi) {
  for (int32_t run = 0; run < runs; run++) {
    T *__restrict q = p + run * run_stride;
    for (int32_t i = 0; i < tiles; i++, q += L) {
      if constexpr (L == 1) {
        *q = T(0);
      } else {
        const uint32_t keep_lo = i < partial ? lo : 0;
        const uint32_t keep_hi = i < partial ? hi : 0;
        aie::mask<L> keep;
        if constexpr (L > 32)
          keep = aie::mask<L>::from_uint32(keep_lo, keep_hi);
        else
          keep = aie::mask<L>::from_uint32(keep_lo);
        aie::store_v(q,
                     aie::select(aie::zeros<T, L>(), aie::load_v<L>(q), keep));
      }
    }
  }
}

// Zeroes A's columns and B's rows from `k_valid` on, in the micro-tiles the
// matmul of the same r, s, t reads, so a K tile a reduction ends inside adds
// nothing past its end, whatever the rest of the tile holds.
template <typename T, unsigned m, unsigned k, unsigned n, unsigned r,
          unsigned s, unsigned t, bool b_row_maj>
static inline void matmul_zero_k_tail(T *__restrict pA, T *__restrict pB,
                                      int32_t k_valid) {
  const int32_t first = k_valid / s;
  const int32_t part = k_valid % s;
  // The lanes of a 32-lane word whose column, `s` to a row, is below `part`.
  uint32_t columns = 0;
  for (unsigned i = 0; i < 32; i += s)
    columns |= ((1u << part) - 1) << i;
  zero_micro_tiles<T, r * s>(pA + first * r * s, m / r, (k / s) * r * s,
                             k / s - first, 1, columns, columns);
  if constexpr (b_row_maj) {
    const int32_t rows = part * t;
    zero_micro_tiles<T, s * t>(pB + first * (n / t) * s * t, 1, 0,
                               (k / s - first) * (n / t), n / t,
                               rows >= 32 ? ~0u : (1u << rows) - 1,
                               rows > 32 ? (1u << (rows - 32)) - 1 : 0);
  } else {
    zero_micro_tiles<T, t * s>(pB + first * t * s, n / t, (k / s) * t * s,
                               k / s - first, 1, columns, columns);
  }
}

extern "C" {

#define matmul_k_tail_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_##mlir_type_in##_##mlir_type_out##_k_tail(                       \
      ctype_in *a_in, ctype_in *b_in, int32_t k_valid) {                       \
    matmul_zero_k_tail<ctype_in, DIM_M, DIM_K, DIM_N, r, s, t, is_b_row_maj>(  \
        a_in, b_in, k_valid);                                                  \
  }

#define matmul_scalar_k_tail_c_func(ctype_in, mlir_type_in, ctype_out,         \
                                    mlir_type_out, r, s, t)                    \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out##_k_tail(                \
      ctype_in *a_in, ctype_in *b_in, int32_t k_valid) {                       \
    matmul_zero_k_tail<ctype_in, DIM_M, DIM_K, DIM_N, 1, 1, 1, is_b_row_maj>(  \
        a_in, b_in, k_valid);                                                  \
  }

#ifndef SCALAR_ONLY
combos(matmul_k_tail_c_func)
#endif
#ifndef VECTORIZED_ONLY
    combos(matmul_scalar_k_tail_c_func)
#endif

} // extern "C"

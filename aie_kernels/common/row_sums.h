//===- row_sums.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_ROW_SUMS_H
#define AIE_KERNELS_COMMON_ROW_SUMS_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"

// v and w hold rows of L partial sums each. Adds lane i + L / 2 onto lane i of
// every row, the pairing reduce_add uses, and returns v's rows then w's.
template <unsigned L, unsigned N>
static inline aie::vector<float, N> fold2(aie::vector<float, N> v,
                                          aie::vector<float, N> w) {
  auto [lo, hi] = aie::interleave_unzip(v, w, L / 2);
  return aie::add(lo, hi);
}

// Sums each row of L lanes in v into one lane. A 16-lane vector folds against
// itself, since narrower unzips lower to per-lane extract and insert.
template <unsigned L, unsigned N>
static inline aie::vector<float, 16> fold_rows(aie::vector<float, N> v) {
  if constexpr (L == 1)
    return v.template grow_extract<16>(0);
  else if constexpr (N == 16)
    return fold_rows<L / 2, 16>(fold2<L>(v, v).template extract<16>(0));
  else
    return fold_rows<L / 2, N / 2>(
        fold2<L>(v.template extract<N / 2>(0), v.template extract<N / 2>(1)));
}

// Packs four rows' accumulators into one, quarter q holding row q's 16 partial
// sums, halving each row the way reduce_add does. Down to 16 lanes a halving
// is a permutation of accumulator quarters, so it needs no shuffle.
template <uint32_t r>
static inline aie::accum<accfloat, 64>
pack_rows4(aie::accum<accfloat, r> a0, aie::accum<accfloat, r> a1,
           aie::accum<accfloat, r> a2, aie::accum<accfloat, r> a3) {
  if constexpr (r > 64) {
    auto half = [](aie::accum<accfloat, r> x) {
      return aie::add(x.template extract<r / 2>(0),
                      x.template extract<r / 2>(1));
    };
    return pack_rows4<r / 2>(half(a0), half(a1), half(a2), half(a3));
  } else if constexpr (r == 16) {
    return aie::concat(a0, a1, a2, a3);
  } else {
    static_assert(r == 32 || r == 64);
    // Two rows of 32 lanes per accumulator.
    aie::accum<accfloat, 64> t01, t23;
    if constexpr (r == 64) {
      auto pair = [](aie::accum<accfloat, 64> x, aie::accum<accfloat, 64> y) {
        return aie::add(
            aie::concat(x.template extract<32>(0), y.template extract<32>(0)),
            aie::concat(x.template extract<32>(1), y.template extract<32>(1)));
      };
      t01 = pair(a0, a1);
      t23 = pair(a2, a3);
    } else {
      t01 = aie::concat(a0, a1);
      t23 = aie::concat(a2, a3);
    }
    return aie::add(
        aie::concat(t01.template extract<16>(0), t01.template extract<16>(2),
                    t23.template extract<16>(0), t23.template extract<16>(2)),
        aie::concat(t01.template extract<16>(1), t01.template extract<16>(3),
                    t23.template extract<16>(1), t23.template extract<16>(3)));
  }
}

// Four rows of A times b, packed by pack_rows4. Each row walks its own cursor:
// one pointer stepped through all four rows chains every load on the previous
// load's post-increment. The first chunk multiplies instead of accumulating
// onto zeros, which would be reloaded from the stack for every group. Short
// rows unroll fully, so a whole group is one block the scheduler can overlap
// with its neighbor. Not inlined, it returns its accumulator through the stack.
template <uint32_t r, uint32_t k>
__attribute__((always_inline)) static inline aie::accum<accfloat, 64>
mac_rows4(const bfloat16 *__restrict a, const bfloat16 *__restrict b) {
  constexpr uint32_t chunks = k / r;
  const bfloat16 *__restrict a0 = a;
  const bfloat16 *__restrict a1 = a + k;
  const bfloat16 *__restrict a2 = a + 2 * k;
  const bfloat16 *__restrict a3 = a + 3 * k;
  const bfloat16 *__restrict pb = b;
  aie::vector<bfloat16, r> b_0 = aie::load_v<r>(pb);
  pb += r;
  aie::accum<accfloat, r> acc0 = aie::mul(aie::load_v<r>(a0), b_0);
  a0 += r;
  aie::accum<accfloat, r> acc1 = aie::mul(aie::load_v<r>(a1), b_0);
  a1 += r;
  aie::accum<accfloat, r> acc2 = aie::mul(aie::load_v<r>(a2), b_0);
  a2 += r;
  aie::accum<accfloat, r> acc3 = aie::mul(aie::load_v<r>(a3), b_0);
  a3 += r;
  auto step = [&]() {
    aie::vector<bfloat16, r> b_vec = aie::load_v<r>(pb);
    pb += r;
    acc0 = aie::mac(acc0, aie::load_v<r>(a0), b_vec);
    a0 += r;
    acc1 = aie::mac(acc1, aie::load_v<r>(a1), b_vec);
    a1 += r;
    acc2 = aie::mac(acc2, aie::load_v<r>(a2), b_vec);
    a2 += r;
    acc3 = aie::mac(acc3, aie::load_v<r>(a3), b_vec);
    a3 += r;
  };
  if constexpr (chunks <= 4) {
    AIE_LOOP_UNROLL_FULL
    for (uint32_t i = 1; i < chunks; i++)
      step();
  } else {
    for (uint32_t i = 1; i < chunks; i++)
      step();
  }
  return pack_rows4<r>(acc0, acc1, acc2, acc3);
}

#endif

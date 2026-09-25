//===- mv_bf16.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// IRON's bf16 GEMV: c[row_offset ..] += A * b over a row-major A, signature
// (m, row_offset, A, b, c). mv_i16.cc is the int16 counterpart, which reads A
// word-transposed. They shared the name mv.cc, in two directories.

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif

#ifndef DIM_K
#error Please define DIM_K at compile time (for example, -DDIM_K=128).
#endif

void matvec_scalar(uint32_t m, uint32_t k, const bfloat16 *__restrict a,
                   const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  for (uint32_t row = 0; row < m; row++) {
    float acc = 0;
    for (uint32_t i = 0; i < k; i++) {
      acc += a[row * k + i] * b[i];
    }
    c[row] = static_cast<bfloat16>(acc);
  }
  ::aie::set_rounding(saved_rounding);
}

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

/*
Matrix-vector multiplication kernel

 - m: Number of output rows == number of rows in the input matrix
 - k: Number of columns in the input matrix == length of the input vector
 - a: Pointer to the input matrix, stored in row-major order
 - b: Pointer to the input vector
 - c: Pointer to the output vector
 - r: Vector size; data from the matrix and vector will be loaded in and
processed in chunks of this size
*/
template <uint32_t r, uint32_t k>
void matvec_vectorized(uint32_t m, const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b, bfloat16 *__restrict c) {
  static_assert(k % r == 0);
  static_assert(k >= r);
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr uint32_t chunks = k / r;

  // Four rows at a time. One b chunk then feeds four macs instead of one, and
  // one transposed tree sums all four rows where reduce_add_v runs four trees
  // side by side. The tree is a latency chain, so each group's finishes in the
  // next iteration, under that group's macs. Behind a mac loop only the store
  // waits: the packed accumulator would spill across the loop. On AIE2 four
  // 64-lane accumulators fill the accumulator file, so the packed one would
  // spill there beside the next group's too.
#if AIE_TUNED_AIE2P
  constexpr bool fold_late = chunks <= 4;
#else
  constexpr bool fold_late = chunks <= 4 && r < 64;
#endif
  auto defer = [](aie::accum<accfloat, 64> u) {
    if constexpr (fold_late)
      return u;
    else
      return fold_rows<16>(u.template to_vector<float>());
  };
  auto store4 = [](bfloat16 *__restrict out, auto deferred) {
    aie::vector<float, 16> sums;
    if constexpr (fold_late)
      sums = fold_rows<16>(deferred.template to_vector<float>());
    else
      sums = deferred;
    aie::vector<bfloat16, 16> v =
        aie::accum<accfloat, 16>(sums).template to_vector<bfloat16>();
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
    out[3] = v[3];
  };
  uint32_t groups = m / 4;
  if (groups > 0) {
    auto prev = defer(mac_rows4<r, k>(a, b));
    for (uint32_t g = 1; g < groups; g++, c += 4) {
      a += 4 * k;
      auto next = defer(mac_rows4<r, k>(a, b));
      store4(c, prev);
      prev = next;
    }
    store4(c, prev);
    a += 4 * k;
    c += 4;
  }

  // m need not be a multiple of four.
  for (uint32_t row = groups * 4; row < m; row++, c++) {
    aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
    AIE_LOOP_MIN_ITERATION_COUNT(chunks)
    for (uint32_t i = 0; i < chunks; i++, a += r)
      acc = aie::mac(acc, aie::load_v<r>(a), aie::load_v<r>(b + i * r));
    *c =
        static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
  }
  ::aie::set_rounding(saved_rounding);
}

extern "C" {

/* The row offset parameter in the functions below is a workaround. The output
 * will be written to c + row_offset * m. This is simpler than to do pointer
 * arithmetic in the calling MLIR code, but that's all this is for -- an offset
 * into `c`.  */

void matvec_scalar_bf16_bf16(uint32_t m, uint32_t row_offset,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             bfloat16 *__restrict c_out) {
  event0();
  c_out += row_offset;
  matvec_scalar(m, DIM_K, a_in, b_in, c_out);
  event1();
}

void matvec_vectorized_bf16_bf16(uint32_t m, uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out) {
  event0();
  c_out += row_offset;
  matvec_vectorized<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
  event1();
}

} // extern "C"

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
//
// Built with -DA_COL_MAJ, the file exports the column-major variant instead:
// A stored (K, M), the transpose, a chunk of its stored rows per call. It
// sums in the row-major kernel's order, operation for operation, so the two
// layouts give the same bits (see matvec_vectorized_col_maj).
//
// Built with -DA_CODED, it exports the coded-rows variant instead: A arrives
// as aie.utils.coded rows, which the core decodes into a scratch tile for
// the row-major kernel, so it returns that kernel's bits.

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#include "../common/row_sums.h"

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif

#ifndef DIM_K
#error Please define DIM_K at compile time (for example, -DDIM_K=128).
#endif

#ifndef A_COL_MAJ

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
#if AIE_TUNED_AIE2
    // mac_elem_16_2 sums lanes i and i + 16, so each accumulator folds 32
    // products per chunk.
    if constexpr (r % 64 == 0 && chunks > 1) {
      constexpr uint32_t P = r / 32;
      v16accfloat acc[P];
      AIE_LOOP_UNROLL_FULL
      for (uint32_t q = 0; q < P; q++)
        acc[q] = mul_elem_16_2(aie::load_v<32>(a + 32 * q).to_native(),
                               aie::load_v<32>(b + 32 * q).to_native());
      a += r;
      const bfloat16 *__restrict pb = b + r;
      AIE_LOOP_MIN_ITERATION_COUNT(chunks - 1)
      for (uint32_t i = 1; i < chunks; i++, a += r, pb += r) {
        AIE_LOOP_UNROLL_FULL
        for (uint32_t q = 0; q < P; q++)
          acc[q] =
              mac_elem_16_2(aie::load_v<32>(a + 32 * q).to_native(),
                            aie::load_v<32>(pb + 32 * q).to_native(), acc[q]);
      }
      aie::accum<accfloat, 16> sum(acc[0]);
      AIE_LOOP_UNROLL_FULL
      for (uint32_t q = 1; q < P; q++)
        sum = aie::add(sum, aie::accum<accfloat, 16>(acc[q]));
      *c = static_cast<bfloat16>(aie::reduce_add(sum.to_vector<float>()));
      continue;
    }
#endif
    aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
    AIE_LOOP_MIN_ITERATION_COUNT(chunks)
    for (uint32_t i = 0; i < chunks; i++, a += r)
      acc = aie::mac(acc, aie::load_v<r>(a), aie::load_v<r>(b + i * r));
    *c =
        static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
  }
  ::aie::set_rounding(saved_rounding);
}

#ifdef A_CODED

#ifndef CAPACITY
#define CAPACITY 15
#endif

#include "../common/coded_rows.h"

extern "C" {

// a_scratch holds the decoded rows, which feed the unchanged matvec, so the
// result is plain mv's on the decoded matrix, bit for bit.
void matvec_coded_bf16_bf16(uint32_t m, uint32_t row_offset,
                            const uint8_t *__restrict a_coded,
                            bfloat16 *__restrict a_scratch,
                            const bfloat16 *__restrict b_in,
                            bfloat16 *__restrict c_out) {
  event0();
  c_out += row_offset;
  decode_coded_rows<DIM_K, CAPACITY>(m, a_coded, a_scratch);
  matvec_vectorized<VEC_SIZE, DIM_K>(m, a_scratch, b_in, c_out);
  event1();
}

} // extern "C"

#else // A_CODED

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

#endif // A_CODED

#else // A_COL_MAJ

#ifndef DIM_M
#error Please define DIM_M, the outputs per call, at compile time.
#endif

// The flags argument: FIRST starts the partial sums (acc is not read), LAST
// finishes them into c. One call with both takes a whole K.
#define MV_COL_MAJ_FIRST 1
#define MV_COL_MAJ_LAST 2

// matvec_vectorized<r, K> sums lane j of a row over positions r * i + j in
// order (mul, then macs), then halves: lane j += lane j + r / 2, ... 1. With
// A stored (K, M), lane j of all m outputs is row j of acc, and position t is
// one mac of stored row t by b[t] into row t % r; a halving adds whole rows.
// Same float32 ops on the same operands, so the same bits, on rows the
// row-major kernel computes in groups of four (its tail uses reduce_add).
// acc carries across calls, so K is any number of k-row chunks.
//  - k: stored rows per call, a multiple of r; m: 16, 32 or a multiple of 64
//  - acc: r * m floats, lane-major; c: m outputs, written by the LAST call
template <uint32_t r, uint32_t k, uint32_t m, bool first>
static inline void accumulate_col_maj(const bfloat16 *__restrict a,
                                      const bfloat16 *__restrict b,
                                      float *__restrict acc) {
  constexpr uint32_t n = m < 64 ? m : 64;
  constexpr uint32_t rounds = k / r;
  for (uint32_t j = 0; j < r; j++) {
    for (uint32_t v = 0; v < m; v += n) {
      const bfloat16 *__restrict aj = a + j * m + v;
      float *__restrict row = acc + j * m + v;
      aie::accum<accfloat, n> sum;
      uint32_t i = 0;
      if constexpr (first) {
        sum = aie::mul(aie::load_v<n>(aj), aie::broadcast<bfloat16, n>(b[j]));
        i = 1;
      } else {
        sum.from_vector(aie::load_v<n>(row));
      }
      AIE_LOOP_UNROLL_FULL
      for (; i < rounds; i++)
        sum = aie::mac(sum, aie::load_v<n>(aj + i * r * m),
                       aie::broadcast<bfloat16, n>(b[i * r + j]));
      aie::store_v(row, sum.template to_vector<float>());
    }
  }
}

// The row-major kernel's halvings over the rows of acc, then its rounding.
template <uint32_t r, uint32_t m>
static inline void finish_col_maj(float *__restrict acc,
                                  bfloat16 *__restrict c) {
  constexpr uint32_t n = m < 64 ? m : 64;
  for (uint32_t half = r / 2; half >= 1; half /= 2) {
    for (uint32_t j = 0; j < half; j++) {
      for (uint32_t v = 0; v < m; v += n) {
        float *__restrict lo = acc + j * m + v;
        const float *__restrict hi = acc + (j + half) * m + v;
        if (half >= 16) {
          aie::accum<accfloat, n> x, y;
          x.from_vector(aie::load_v<n>(lo));
          y.from_vector(aie::load_v<n>(hi));
          aie::store_v(lo, aie::add(x, y).template to_vector<float>());
        } else {
          aie::store_v(lo, aie::add(aie::load_v<n>(lo), aie::load_v<n>(hi)));
        }
      }
    }
  }
  for (uint32_t v = 0; v < m; v += n) {
    aie::accum<accfloat, n> sum;
    sum.from_vector(aie::load_v<n>(acc + v));
    aie::store_v(c + v, sum.template to_vector<bfloat16>());
  }
}

template <uint32_t r, uint32_t k, uint32_t m>
void matvec_vectorized_col_maj(uint32_t flags, const bfloat16 *__restrict a,
                               const bfloat16 *__restrict b,
                               float *__restrict acc, bfloat16 *__restrict c) {
  constexpr uint32_t n = m < 64 ? m : 64;
  static_assert(k > 0 && k % r == 0,
                "a call takes one or more whole rounds of the r lanes");
  static_assert(n % 16 == 0 && m % n == 0,
                "outputs come in vectors of 16, 32 or 64");
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  if (flags & MV_COL_MAJ_FIRST)
    accumulate_col_maj<r, k, m, true>(a, b, acc);
  else
    accumulate_col_maj<r, k, m, false>(a, b, acc);
  if (flags & MV_COL_MAJ_LAST)
    finish_col_maj<r, m>(acc, c);
  ::aie::set_rounding(saved_rounding);
}

extern "C" {

void matvec_vectorized_col_maj_bf16_bf16(uint32_t flags,
                                         const bfloat16 *__restrict a_in,
                                         const bfloat16 *__restrict b_in,
                                         float *__restrict acc,
                                         bfloat16 *__restrict c_out) {
  event0();
  matvec_vectorized_col_maj<VEC_SIZE, DIM_K, DIM_M>(flags, a_in, b_in, acc,
                                                    c_out);
  event1();
}

} // extern "C"

#endif // A_COL_MAJ

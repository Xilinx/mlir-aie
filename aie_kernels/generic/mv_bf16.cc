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
  // reduce_add_v folds the four accumulators in a single pass where four
  // separate reduction trees used to run -- the reduction, not the mac, is
  // what a short row costs.
  uint32_t row = 0;
  for (; row + 4 <= m; row += 4, a += 4 * k, c += 4) {
    aie::accum<accfloat, r> acc0 = aie::zeros<accfloat, r>();
    aie::accum<accfloat, r> acc1 = acc0;
    aie::accum<accfloat, r> acc2 = acc0;
    aie::accum<accfloat, r> acc3 = acc0;
    const bfloat16 *__restrict pa = a;
    AIE_LOOP_MIN_ITERATION_COUNT(chunks)
    for (uint32_t i = 0; i < chunks; i++, pa += r) {
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b + i * r);
      acc0 = aie::mac(acc0, aie::load_v<r>(pa), b_vec);
      acc1 = aie::mac(acc1, aie::load_v<r>(pa + k), b_vec);
      acc2 = aie::mac(acc2, aie::load_v<r>(pa + 2 * k), b_vec);
      acc3 = aie::mac(acc3, aie::load_v<r>(pa + 3 * k), b_vec);
    }
    aie::vector<float, r> sums = aie::reduce_add_v(
        acc0.template to_vector<float>(), acc1.template to_vector<float>(),
        acc2.template to_vector<float>(), acc3.template to_vector<float>());
    c[0] = static_cast<bfloat16>(sums[0]);
    c[1] = static_cast<bfloat16>(sums[1]);
    c[2] = static_cast<bfloat16>(sums[2]);
    c[3] = static_cast<bfloat16>(sums[3]);
  }

  // m need not be a multiple of four.
  for (; row < m; row++, c++) {
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
  c_out += row_offset;
  matvec_scalar(m, DIM_K, a_in, b_in, c_out);
}

void matvec_vectorized_bf16_bf16(uint32_t m, uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_vectorized<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
}

} // extern "C"

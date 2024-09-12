//===- mm_b_col_maj.cc ------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contins a matrix-multiplication microkernel that operates on a
// transposed B matrix (or, equivalently, a B matrix that is stored in column-
// major format). A remains row-major.

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_vectorized_b_col_maj(const T_in *__restrict pA,
                                               const T_in *__restrict pB,
                                               T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accfloat>;

  event0();

  // For int16 (4x4x4), this implementation iterates over the output space in
  // steps of 4x4 tiles; each iteration makes an r*s, s*t and r*t step in the
  // input and output space, respectively. The data layout expected is such
  // that each r*s/s*t/r*t tile's elements are laid out contiguously in
  // row-major order, and tiles themselves are organized in row-major
  // order. For example, for 4x4x4 tiles, this means that an element in
  // row 1, column 0 would be stored at offset 4 (since the first 4x4 tile
  // is laid out contiguously in row-major). An element in row 0, column 4
  // would be stored at offset 16 in the same example.

  for (unsigned z = 0; z < rowA; z += 2)
    chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
        // chess_loop_range(2, ) {
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (j * colA) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + ((j + 1) * colA) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 =
              aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
          pB1 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B1 =
              aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
          pB2 += MMUL::size_B;

          // We modify the library documentation implementation to accumulate
          // in the C dimension, since this vectorized kernel will be called
          // multiple times as we further tile the input at a higher level.
          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          for (unsigned i = 1; i < colA; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              // chess_unroll_loop() {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
              pB1 += MMUL::size_B;
              B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
              pB2 += MMUL::size_B;
              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
        }
    }

  event1();
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_f32_b_col_maj(const bfloat16 *__restrict pA,
                                           const bfloat16 *__restrict pB,
                                           float *__restrict pC) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized_b_col_maj<bfloat16, float, m / r, k / s, n / t, r, s,
                                     t>(pA, pB, pC);
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#define combos(X) X(bfloat16, bf16, float, f32, 4, 8, 4)

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,                   \
                                 mlir_type_out, r, s, t)                              \
  void matmul_##mlir_type_in##_##mlir_type_out##_b_col_maj(                           \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                             \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out##_b_col_maj< \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);                                      \
  }

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

combos(matmul_vectorized_c_func) combos(zero_vectorized_c_func)
    combos(zero_scalar_c_func)

} // extern "C"

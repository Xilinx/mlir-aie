//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

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

template <typename T_in, typename T_out, int M, int K, int N>
void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < K; i++) {
        running_sum += a[row * K + i] * b[i * N + col];
      }
      c[row * N + col] += running_sum;
    }
  }
  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized(const T_in *__restrict pA, const T_in *__restrict pB,
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
        chess_loop_range(2, ) {
          const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B * colB;

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
            chess_prepare_for_pipelining chess_loop_range(3, ) {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B * colB;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B * colB;

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
void matmul_vectorized_4x4x4_i16_i16(const int16 *__restrict pA,
                                     const int16 *__restrict pB,
                                     int16 *__restrict pC) {
  // matmul_vectorized operates on two 4x4 input blocks of A, and two 4x4 input
  // blocks of B in each iteration. Make sure we have at least 2 blocks in each
  // dimension, and that our input matrix is evenly divisible.
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<int16, int16, m / r, k / s, n / t, r, s, t>(pA, pB,
                                                                       pC);
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                                       const bfloat16 *__restrict pB,
                                       bfloat16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<bfloat16, bfloat16, m / r, k / s, n / t, r, s, t>(
      pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x8x4_bf16_f32(const bfloat16 *__restrict pA,
                                      const bfloat16 *__restrict pB,
                                      float *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<bfloat16, float, m / r, k / s, n / t, r, s, t>(
      pA, pB, pC);
}

extern "C" {

#define combos(X)                                                              \
  X(int16, i16, int16, i16, 4, 4, 4)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)                                   \
  X(bfloat16, bf16, float, f32, 4, 8, 4)

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void matmul_##mlir_type_in##_##mlir_type_out(ctype_in *a_in, ctype_in *b_in, \
                                               ctype_out *c_out) {             \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        64, 32, 64>(a_in, b_in, c_out);                                        \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar<ctype_in, ctype_out, 64, 32, 64>(a_in, b_in, c_out);         \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out) {                                \
    zero_vectorized<ctype_out, 64, 64, 32>(c_out);                             \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t)                                            \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, 64, 64>(c_out);                                     \
  }

combos(matmul_vectorized_c_func) combos(matmul_scalar_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"

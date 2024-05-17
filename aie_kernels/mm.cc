//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
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

// Suppose A is a 64x64 tensor and B is a 64x64 tensor, and r=4, s=8, t=4.
//
// Let A[i,j] be the element at row i and column j of A, and
//     B[i,j] be the element at row i and column j of B.
//
// The expectations that this implementation makes are:
//
// 1) all elements of A are contiguous in memory, starting from pA + offsetA
// 2) all elements of B are contiguous in memory, starting from pB + offsetB
// 3) all elements of C are contiguous in memory, starting from pC + offsetC
// 4) element A[i,j] is at pA[offsetA + i*8 + (64*8)*(j/8) + j%8]
// 5) element B[i,j] is at pB[offsetB + i*4 + (64*4)*(j/4) + j%4]
//
// 4) and 5) describe vertical stripes of A and B being stored contiguously,
// with a row-major order within each stripe. i.e. A looks like
//
// [A[0,0], ..., A[0,7], A[1,0], ..., A[1,7], A[2,0], ..., A[2,7], ... A[63,0],
// ..., A[63,7], A[0,8], ..., A[0,15], ..., A[63, 64]]
//
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized(const T_in *__restrict pA, unsigned offsetA,
                       const T_in *__restrict pB, unsigned offsetB,
                       T_out *__restrict pC, unsigned offsetC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accfloat>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + offsetC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + offsetC + ((z + 1)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          const T_in *__restrict pA1 = pA + offsetA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + offsetA + ((z + 1)) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + offsetB + (j)*colA * MMUL::size_B;
          const T_in *__restrict pB2 =
              pB + offsetB + ((j + 1)) * colA * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);

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
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;
              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
        }
    }

  event1();
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                                       unsigned offsetA,
                                       const bfloat16 *__restrict pB,
                                       unsigned offsetB,
                                       bfloat16 *__restrict pC,
                                       unsigned offsetC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<bfloat16, bfloat16, m / r, k / s, n / t, r, s, t>(
      pA, offsetA, pB, offsetB, pC, offsetC);
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x8x4_bf16_f32(const bfloat16 *__restrict pA,
                                      unsigned offsetA,
                                      const bfloat16 *__restrict pB,
                                      unsigned offsetB, float *__restrict pC,
                                      unsigned offsetC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<bfloat16, float, m / r, k / s, n / t, r, s, t>(
      pA, offsetA, pB, offsetB, pC, offsetC);
}

extern "C" {

#define combos(X)                                                              \
  X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)                                   \
  X(bfloat16, bf16, float, f32, 4, 8, 4)

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void matmul_##mlir_type_in##_##mlir_type_out(                                \
      ctype_in *a_in, unsigned offsetA, ctype_in *b_in, unsigned offsetB,      \
      ctype_out *c_out, unsigned offsetC) {                                    \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        64, 64, 64>(a_in, offsetA, b_in, offsetB, c_out, offsetC);             \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out, unsigned offsetC) {              \
    zero_vectorized<ctype_out, 64, 64, 32>(c_out, offsetC);                    \
  }

combos(matmul_vectorized_c_func) combos(zero_vectorized_c_func)

} // extern "C"

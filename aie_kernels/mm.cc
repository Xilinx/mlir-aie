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

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized(const T_in *__restrict pA, unsigned offsetA,
                       const T_in *__restrict pB, unsigned offsetB,
                       T_out *__restrict pC, unsigned offsetC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accfloat>;

  event0();

  for (unsigned z = 0; z < rowA; z += 4)
    chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + offsetC + (z)*MMUL::size_C;
      T_out *__restrict pC2 = pC + offsetC + ((z + 1)) * MMUL::size_C;
      T_out *__restrict pC3 = pC + offsetC + ((z + 2)) * MMUL::size_C;
      T_out *__restrict pC4 = pC + offsetC + ((z + 3)) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4)
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          const T_in *__restrict pA1 = pA + offsetA + (z)*MMUL::size_A;
          const T_in *__restrict pA2 = pA + offsetA + ((z + 1)) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + offsetA + ((z + 2)) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + offsetA + ((z + 3)) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + offsetB + (j)*MMUL::size_B;
          const T_in *__restrict pB2 = pB + offsetB + ((j + 1)) * MMUL::size_B;
          const T_in *__restrict pB3 = pB + offsetB + ((j + 2)) * MMUL::size_B;
          const T_in *__restrict pB4 = pB + offsetB + ((j + 3)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A3 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += rowA * MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB3);
          pB3 += MMUL::size_B;
          aie::vector<T_in, MMUL::size_B> B3 = aie::load_v<MMUL::size_B>(pB4);
          pB4 += MMUL::size_B;

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C02 =
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C03 =
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C12 =
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C13 =
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C22 =
              aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C23 =
              aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C * rowA);

          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C32 =
              aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C * rowA);
          aie::vector<T_out, MMUL::size_C> acc_C33 =
              aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C * rowA);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C02(acc_C02);
          MMUL C03(acc_C03);

          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C12(acc_C12);
          MMUL C13(acc_C13);

          MMUL C20(acc_C20);
          MMUL C21(acc_C21);
          MMUL C22(acc_C22);
          MMUL C23(acc_C23);

          MMUL C30(acc_C30);
          MMUL C31(acc_C31);
          MMUL C32(acc_C32);
          MMUL C33(acc_C33);

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);

          C02.mac(A0, B2);
          C03.mac(A0, B3);
          C12.mac(A1, B2);
          C13.mac(A1, B3);

          C20.mac(A2, B0);
          C21.mac(A2, B1);
          C30.mac(A3, B0);
          C31.mac(A3, B1);

          C22.mac(A2, B2);
          C23.mac(A2, B3);
          C32.mac(A3, B2);
          C33.mac(A3, B3);

          for (unsigned i = 1; i < colA; ++i)
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += rowA * MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += rowA * MMUL::size_A;
              A2 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += rowA * MMUL::size_A;
              A3 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += rowA * MMUL::size_A;

              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B;
              B2 = aie::load_v<MMUL::size_B>(pB3);
              pB3 += MMUL::size_B;
              B3 = aie::load_v<MMUL::size_B>(pB4);
              pB4 += MMUL::size_B;

              C00.mac(A0, B0);
              C01.mac(A0, B1);
              C10.mac(A1, B0);
              C11.mac(A1, B1);

              C02.mac(A0, B2);
              C03.mac(A0, B3);
              C12.mac(A1, B2);
              C13.mac(A1, B3);

              C20.mac(A2, B0);
              C21.mac(A2, B1);
              C30.mac(A3, B0);
              C31.mac(A3, B1);

              C22.mac(A2, B2);
              C23.mac(A2, B3);
              C32.mac(A3, B2);
              C33.mac(A3, B3);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C02.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;
          aie::store_v(pC1, C03.template to_vector<T_out>());
          pC1 += MMUL::size_C * rowA;

          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C12.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;
          aie::store_v(pC2, C13.template to_vector<T_out>());
          pC2 += MMUL::size_C * rowA;

          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C22.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;
          aie::store_v(pC3, C23.template to_vector<T_out>());
          pC3 += MMUL::size_C * rowA;

          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C32.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
          aie::store_v(pC4, C33.template to_vector<T_out>());
          pC4 += MMUL::size_C * rowA;
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

combos(matmul_vectorized_c_func)

} // extern "C"

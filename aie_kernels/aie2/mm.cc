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

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      c[row * colB + col] += running_sum;
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
        // chess_loop_range(2, ) {
        chess_prepare_for_pipelining chess_loop_range(8, ) {
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
            chess_prepare_for_pipelining chess_loop_range(7, ) {
              // chess_unroll_loop() {
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

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_2x2(const T_in *__restrict pA, const T_in *__restrict pB,
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

  for (unsigned z = 0; z < rowA; z += 4)
    chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC3 = pC + ((z + 2) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC4 = pC + ((z + 3) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 4)
        // chess_loop_range(2, ) {
        chess_prepare_for_pipelining chess_loop_range(8, ) {
          const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;
          const T_in *__restrict pB3 = pB + (0 * colB + (j + 2)) * MMUL::size_B;
          const T_in *__restrict pB4 = pB + (0 * colB + (j + 3)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A2 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A3 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B2 = aie::load_v<MMUL::size_B>(pB3);
          pB3 += MMUL::size_B * colB;
          aie::vector<T_in, MMUL::size_B> B3 = aie::load_v<MMUL::size_B>(pB4);
          pB4 += MMUL::size_B * colB;

          // We modify the library documentation implementation to accumulate
          // in the C dimension, since this vectorized kernel will be called
          // multiple times as we further tile the input at a higher level.
          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C02 =
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C03 =
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C12 =
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C13 =
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C);

          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C22 =
              aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C23 =
              aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C);

          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C32 =
              aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C33 =
              aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C);

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
              // chess_unroll_loop() {
              A0 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A1 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              A2 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += MMUL::size_A;
              A3 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += MMUL::size_A;

              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B * colB;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B * colB;
              B2 = aie::load_v<MMUL::size_B>(pB3);
              pB3 += MMUL::size_B * colB;
              B3 = aie::load_v<MMUL::size_B>(pB4);
              pB4 += MMUL::size_B * colB;

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
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C02.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C03.template to_vector<T_out>());
          pC1 += MMUL::size_C;

          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C12.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C13.template to_vector<T_out>());
          pC2 += MMUL::size_C;

          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C22.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C23.template to_vector<T_out>());
          pC3 += MMUL::size_C;

          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C32.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C33.template to_vector<T_out>());
          pC4 += MMUL::size_C;
        }
    }

  event1();
}

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_1x2(const T_in *__restrict pA, const T_in *__restrict pB,
                           T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in, accfloat>;
  unsigned long long time;
  event0();

  // Microkernel extended to maximize accumulator usage

  // unsigned long long start = get_cycles ();
  for (unsigned z = 0; z < rowA; z += 4)
    chess_loop_range(2, ) {
      T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC3 = pC + ((z + 2) * colB + 0) * MMUL::size_C;
      T_out *__restrict pC4 = pC + ((z + 3) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
        chess_loop_range(2, ) {
          const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;

          const T_in *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
          const T_in *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;

          aie::vector<T_in, MMUL::size_A> A01 = aie::load_v<MMUL::size_A>(pA1);
          pA1 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A11 = aie::load_v<MMUL::size_A>(pA2);
          pA2 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A21 = aie::load_v<MMUL::size_A>(pA3);
          pA3 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_A> A31 = aie::load_v<MMUL::size_A>(pA4);
          pA4 += MMUL::size_A;
          aie::vector<T_in, MMUL::size_B> B01 = aie::load_v<MMUL::size_B>(pB1);
          pB1 += (MMUL::size_B * colB);
          aie::vector<T_in, MMUL::size_B> B11 = aie::load_v<MMUL::size_B>(pB2);
          pB2 += (MMUL::size_B * colB);

          aie::vector<T_out, MMUL::size_C> acc_C00 =
              aie::load_v<MMUL::size_C>(pC1);
          aie::vector<T_out, MMUL::size_C> acc_C01 =
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C10 =
              aie::load_v<MMUL::size_C>(pC2);
          aie::vector<T_out, MMUL::size_C> acc_C11 =
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C20 =
              aie::load_v<MMUL::size_C>(pC3);
          aie::vector<T_out, MMUL::size_C> acc_C21 =
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          aie::vector<T_out, MMUL::size_C> acc_C30 =
              aie::load_v<MMUL::size_C>(pC4);
          aie::vector<T_out, MMUL::size_C> acc_C31 =
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);

          MMUL C00(acc_C00);
          MMUL C01(acc_C01);
          MMUL C10(acc_C10);
          MMUL C11(acc_C11);
          MMUL C20(acc_C20);
          MMUL C21(acc_C21);
          MMUL C30(acc_C30);
          MMUL C31(acc_C31);

          C00.mac(A01, B01);
          C01.mac(A01, B11);
          C10.mac(A11, B01);
          C11.mac(A11, B11);
          C20.mac(A21, B01);
          C21.mac(A21, B11);
          C30.mac(A31, B01);
          C31.mac(A31, B11);

          for (unsigned i = 1; i < colA; i += 1)
            chess_prepare_for_pipelining chess_loop_range(3, ) {

              A01 = aie::load_v<MMUL::size_A>(pA1);
              pA1 += MMUL::size_A;
              A11 = aie::load_v<MMUL::size_A>(pA2);
              pA2 += MMUL::size_A;
              A21 = aie::load_v<MMUL::size_A>(pA3);
              pA3 += MMUL::size_A;
              A31 = aie::load_v<MMUL::size_A>(pA4);
              pA4 += MMUL::size_A;
              B01 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += (MMUL::size_B * colB);
              B11 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += (MMUL::size_B * colB);

              C00.mac(A01, B01);
              C01.mac(A01, B11);
              C10.mac(A11, B01);
              C11.mac(A11, B11);
              C20.mac(A21, B01);
              C21.mac(A21, B11);
              C30.mac(A31, B01);
              C31.mac(A31, B11);
            }

          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC3, C20.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC3, C21.template to_vector<T_out>());
          pC3 += MMUL::size_C;
          aie::store_v(pC4, C30.template to_vector<T_out>());
          pC4 += MMUL::size_C;
          aie::store_v(pC4, C31.template to_vector<T_out>());
          pC4 += MMUL::size_C;
        }
    }
  // unsigned long long end = get_cycles ();
  // time = end - start;
  // *pC = (int)time;
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
  // return matmul_vectorized<int16, int16, m / r, k / s, n / t, r, s, t>(pA,
  // pB,
  //                                                                      pC);
  return matmul_vectorized_1x2<int16, int16, m / r, k / s, n / t, r, s, t>(
      pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
void matmul_vectorized_4x4x4_i16_i32(const int16 *__restrict pA,
                                     const int16 *__restrict pB,
                                     int32 *__restrict pC) {
  // matmul_vectorized operates on two 4x4 input blocks of A, and two 4x4 input
  // blocks of B in each iteration. Make sure we have at least 2 blocks in each
  // dimension, and that our input matrix is evenly divisible.
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 4;
  static_assert(m % (2 * r) == 0 && m / (2 * r) > 0);
  static_assert(k % (2 * s) == 0 && k / (2 * s) > 0);
  static_assert(n % (2 * t) == 0 && n / (2 * t) > 0);
  return matmul_vectorized<int16, int32, m / r, k / s, n / t, r, s, t>(pA, pB,
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
  // return matmul_vectorized<bfloat16, bfloat16, m / r, k / s, n / t, r, s, t>(
  return matmul_vectorized_2x2<bfloat16, bfloat16, m / r, k / s, n / t, r, s,
                               t>(pA, pB, pC);
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

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#define combos(X)                                                              \
  X(int16, i16, int16, i16, 4, 4, 4)                                           \
  X(int16, i16, int32, i32, 4, 4, 4)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)                                   \
  X(bfloat16, bf16, float, f32, 4, 8, 4)

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, r, s, t)                       \
  void matmul_##mlir_type_in##_##mlir_type_out(ctype_in *a_in, ctype_in *b_in, \
                                               ctype_out *c_out) {             \
    matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<      \
        DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);                               \
  }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             r, s, t)                                          \
  void matmul_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matmul_scalar<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(a_in, b_in,        \
                                                            c_out);            \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, r, s, t)                         \
  void zero_##mlir_type_out(ctype_out *c_out) {                                \
    zero_vectorized<ctype_out, DIM_M, DIM_N, 32>(c_out);                       \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           r, s, t)                                            \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, DIM_M, DIM_N>(c_out);                               \
  }

combos(matmul_vectorized_c_func) combos(matmul_scalar_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"

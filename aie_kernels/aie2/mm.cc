//===- mm.cc ----------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in, typename T_out, int rowA, int colA, int colB,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        T_in a_val = a[row * colA + i];
        T_in b_val;
        if constexpr (b_row_maj) {
          b_val = b[i * colB + col];
        } else {
          b_val = b[i + col * colA];
        }
        running_sum += a_val * b_val;
      }
      T_out *c_ptr;
      if constexpr (c_row_maj) {
        c_ptr = &c[row * colB + col];
      } else {
        c_ptr = &c[row + col * rowA];
      }
      *c_ptr += running_sum;
    }
  }
  event1();
}

/* Blocked MatMul kernel (vectorized) utilizing the aie::mmul class.
 * The matrices are assumed to be pre-tiled with the following shapes
 * for the aie:mmul class: A => rxs, B => sxt, C => rxt.
 *
 * The matrix dimensions of the kernel are defined by rowA, colA and colB.
 * In this particular kernel we expand the aie::mmul two times in each
 * input matrices A (in 'm' dimension, or rowA) and B (in 'n' dimension, or
 * ColB), leading to a 2x2 expansion in output matrix C (see C00, C01, C10, C11
 * below). This expansion helps with accumulator registers usage, which leads in
 * attaining high kernel efficiency (SIMD utilization).
 *
 * Data within each tile (rxs, sxt and rxt) are assumed to be in row-major
 * order. Also, the entire tiles themselves are stored in row-major order, as
 * shown in the example below for matrix A:
 *
 *      <-s->
 *    _  ________________________
 * 	  r |  1 |  2 |  3 | ...
 * 	  _ |____|____|____|
 * 	    |  x | x+1| x+2| ...
 * 	    |____|____|____|
 * 	    |.
 * 	    |.
 * 	    |.
 *
 * A simplified example of this kernel can be found in the AIE-API
 * documentation: https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned z = 0; z < rowA; z += 2) {
    T_out *__restrict pC1;
    T_out *__restrict pC2;
    if constexpr (c_row_maj) {
      pC1 = pC + (z * colB) * MMUL::size_C;
      pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
    }

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {

        if constexpr (!c_row_maj) {
          pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
          pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
        }
        const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        const T_in *__restrict pB1;
        const T_in *__restrict pB2;
        if constexpr (b_row_maj) {
          pB1 = pB + (j)*MMUL::size_B;
          pB2 = pB + (j + 1) * MMUL::size_B;
        } else {
          pB1 = pB + (j * colA) * MMUL::size_B;
          pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
        }

        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        // Load partial results from C buffer for accumulation in-place. The
        // zero.cc function handles the zeroing of data when a new
        // accumulation is needed (after the 'K' reduction dimension)
        aie::vector<T_out, MMUL::size_C> acc_C00;
        aie::vector<T_out, MMUL::size_C> acc_C01;
        aie::vector<T_out, MMUL::size_C> acc_C10;
        aie::vector<T_out, MMUL::size_C> acc_C11;
        if constexpr (c_row_maj) {
          acc_C00 = aie::load_v<MMUL::size_C>(pC1);
          acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          acc_C10 = aie::load_v<MMUL::size_C>(pC2);
          acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
        } else {
          acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r);
          acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r);
          acc_C10 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r);
          acc_C11 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r);
        }

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A1 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            if constexpr (b_row_maj) {
              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B * colB;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B * colB;
            } else {
              B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
              pB1 += MMUL::size_B;
              B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
              pB2 += MMUL::size_B;
            }

            C00.mac(A0, B0);
            C01.mac(A0, B1);
            C10.mac(A1, B0);
            C11.mac(A1, B1);
          }

        // TODO make shift right here to keep most significat bits
        // when lowering the output
        // example below shows how to shift right 10 bits
        // #define SHIFT 10
        // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

        if constexpr (c_row_maj) {
          aie::store_v(pC1, C00.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC1, C01.template to_vector<T_out>());
          pC1 += MMUL::size_C;
          aie::store_v(pC2, C10.template to_vector<T_out>());
          pC2 += MMUL::size_C;
          aie::store_v(pC2, C11.template to_vector<T_out>());
          pC2 += MMUL::size_C;
        } else {
          aie::store_v(pC1,
                       aie::transpose(C00.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C01.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC1,
                       aie::transpose(C10.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C11.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
        }
      }
  }

  event1();
}

/* Similar to the kernel above, but we expand matrix A (in 'm' dimension, or
 * rowA) 4 times, while matrix B is expanded 2 times (in 'n' dimension, or
 * ColB). This is very helpful in attaining high kernel efficiency for some
 * precisions (e.g., int8)
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_vectorized_4x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned z = 0; z < rowA; z += 4) {
    T_out *__restrict pC1;
    T_out *__restrict pC2;
    T_out *__restrict pC3;
    T_out *__restrict pC4;

    if constexpr (c_row_maj) {
      pC1 = pC + (z * colB + 0) * MMUL::size_C;
      pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
      pC3 = pC + ((z + 2) * colB + 0) * MMUL::size_C;
      pC4 = pC + ((z + 3) * colB + 0) * MMUL::size_C;
    }

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        if constexpr (!c_row_maj) {
          pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
          pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
        }

        const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;

        const T_in *__restrict pB1;
        const T_in *__restrict pB2;
        if constexpr (b_row_maj) {
          pB1 = pB + (j)*MMUL::size_B;
          pB2 = pB + (j + 1) * MMUL::size_B;
        } else {
          pB1 = pB + (j * colA) * MMUL::size_B;
          pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
        }

        aie::vector<T_in, MMUL::size_A> A01;
        aie::vector<T_in, MMUL::size_A> A11;
        aie::vector<T_in, MMUL::size_A> A21;
        aie::vector<T_in, MMUL::size_A> A31;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;

        aie::vector<T_out, MMUL::size_C> acc_C00;
        aie::vector<T_out, MMUL::size_C> acc_C01;
        aie::vector<T_out, MMUL::size_C> acc_C10;
        aie::vector<T_out, MMUL::size_C> acc_C11;
        aie::vector<T_out, MMUL::size_C> acc_C20;
        aie::vector<T_out, MMUL::size_C> acc_C21;
        aie::vector<T_out, MMUL::size_C> acc_C30;
        aie::vector<T_out, MMUL::size_C> acc_C31;

        if constexpr (c_row_maj) {
          acc_C00 = aie::load_v<MMUL::size_C>(pC1);
          acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          acc_C10 = aie::load_v<MMUL::size_C>(pC2);
          acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          acc_C20 = aie::load_v<MMUL::size_C>(pC3);
          acc_C21 = aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          acc_C30 = aie::load_v<MMUL::size_C>(pC4);
          acc_C31 = aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);
        } else {
          acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r);
          acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r);
          acc_C10 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r);
          acc_C11 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r);
          acc_C20 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C), t, r);
          acc_C21 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C), t, r);
          acc_C30 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C), t, r);
          acc_C31 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C), t, r);
        }

        MMUL C00(acc_C00);
        MMUL C01(acc_C01);
        MMUL C10(acc_C10);
        MMUL C11(acc_C11);
        MMUL C20(acc_C20);
        MMUL C21(acc_C21);
        MMUL C30(acc_C30);
        MMUL C31(acc_C31);

        for (unsigned i = 0; i < colA; i += 1)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A01 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A11 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            A21 = aie::load_v<MMUL::size_A>(pA3);
            pA3 += MMUL::size_A;
            A31 = aie::load_v<MMUL::size_A>(pA4);
            pA4 += MMUL::size_A;
            if constexpr (b_row_maj) {
              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += (MMUL::size_B * colB);
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += (MMUL::size_B * colB);
            } else {
              B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
              pB1 += MMUL::size_B;
              B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
              pB2 += MMUL::size_B;
            }

            C00.mac(A01, B0);
            C01.mac(A01, B1);
            C10.mac(A11, B0);
            C11.mac(A11, B1);
            C20.mac(A21, B0);
            C21.mac(A21, B1);
            C30.mac(A31, B0);
            C31.mac(A31, B1);
          }

        if constexpr (c_row_maj) {
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
        } else {
          aie::store_v(pC1,
                       aie::transpose(C00.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C01.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC1,
                       aie::transpose(C10.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C11.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC1,
                       aie::transpose(C20.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C21.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC1,
                       aie::transpose(C30.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C31.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
        }
      }
  }

  event1();
}

/* Similar to the kernel aboves, we expand matrix A (in 'm' dimension, or rowA)
 * 4 times, while matrix B is expanded spatially 4 times (in 'n' dimension, or
 * ColB), for even higher accumulator usage. This is very helpful in attaining
 * high kernel efficiency for some precisions (e.g., bf16)
 */
template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t,
          bool b_row_maj = true, bool c_row_maj = true>
static inline void matmul_vectorized_4x4(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (unsigned z = 0; z < rowA; z += 4) {
    T_out *__restrict pC1;
    T_out *__restrict pC2;
    T_out *__restrict pC3;
    T_out *__restrict pC4;

    if constexpr (c_row_maj) {
      pC1 = pC + (z * colB) * MMUL::size_C;
      pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
      pC3 = pC + ((z + 2) * colB) * MMUL::size_C;
      pC4 = pC + ((z + 3) * colB) * MMUL::size_C;
    }

    for (unsigned j = 0; j < colB; j += 4)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        if constexpr (!c_row_maj) {
          pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
          pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
          pC3 = pC + (j + 2) * rowA * MMUL::size_C + z * MMUL::size_C;
          pC4 = pC + (j + 3) * rowA * MMUL::size_C + z * MMUL::size_C;
        }
        const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA3 = pA + ((z + 2) * colA + 0) * MMUL::size_A;
        const T_in *__restrict pA4 = pA + ((z + 3) * colA + 0) * MMUL::size_A;

        const T_in *__restrict pB1;
        const T_in *__restrict pB2;
        const T_in *__restrict pB3;
        const T_in *__restrict pB4;
        if constexpr (b_row_maj) {
          pB1 = pB + (j)*MMUL::size_B;
          pB2 = pB + (j + 1) * MMUL::size_B;
          pB3 = pB + (j + 2) * MMUL::size_B;
          pB4 = pB + (j + 3) * MMUL::size_B;
        } else {
          pB1 = pB + (j * colA) * MMUL::size_B;
          pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
          pB3 = pB + ((j + 2) * colA) * MMUL::size_B;
          pB4 = pB + ((j + 3) * colA) * MMUL::size_B;
        }

        aie::vector<T_in, MMUL::size_A> A0;
        aie::vector<T_in, MMUL::size_A> A1;
        aie::vector<T_in, MMUL::size_A> A2;
        aie::vector<T_in, MMUL::size_A> A3;
        aie::vector<T_in, MMUL::size_B> B0;
        aie::vector<T_in, MMUL::size_B> B1;
        aie::vector<T_in, MMUL::size_B> B2;
        aie::vector<T_in, MMUL::size_B> B3;

        aie::vector<T_out, MMUL::size_C> acc_C00;
        aie::vector<T_out, MMUL::size_C> acc_C01;
        aie::vector<T_out, MMUL::size_C> acc_C02;
        aie::vector<T_out, MMUL::size_C> acc_C03;

        aie::vector<T_out, MMUL::size_C> acc_C10;
        aie::vector<T_out, MMUL::size_C> acc_C11;
        aie::vector<T_out, MMUL::size_C> acc_C12;
        aie::vector<T_out, MMUL::size_C> acc_C13;

        aie::vector<T_out, MMUL::size_C> acc_C20;
        aie::vector<T_out, MMUL::size_C> acc_C21;
        aie::vector<T_out, MMUL::size_C> acc_C22;
        aie::vector<T_out, MMUL::size_C> acc_C23;

        aie::vector<T_out, MMUL::size_C> acc_C30;
        aie::vector<T_out, MMUL::size_C> acc_C31;
        aie::vector<T_out, MMUL::size_C> acc_C32;
        aie::vector<T_out, MMUL::size_C> acc_C33;

        if constexpr (c_row_maj) {
          acc_C00 = aie::load_v<MMUL::size_C>(pC1);
          acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
          acc_C02 = aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C);
          acc_C03 = aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C);

          acc_C10 = aie::load_v<MMUL::size_C>(pC2);
          acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
          acc_C12 = aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C);
          acc_C13 = aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C);

          acc_C20 = aie::load_v<MMUL::size_C>(pC3);
          acc_C21 = aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C);
          acc_C22 = aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C);
          acc_C23 = aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C);

          acc_C30 = aie::load_v<MMUL::size_C>(pC4);
          acc_C31 = aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C);
          acc_C32 = aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C);
          acc_C33 = aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C);
        } else {
          acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r);
          acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r);
          acc_C02 = aie::transpose(aie::load_v<MMUL::size_C>(pC3), t, r);
          acc_C03 = aie::transpose(aie::load_v<MMUL::size_C>(pC4), t, r);

          acc_C10 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r);
          acc_C11 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r);
          acc_C12 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C), t, r);
          acc_C13 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C), t, r);

          acc_C20 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C), t, r);
          acc_C21 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C), t, r);
          acc_C22 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C), t, r);
          acc_C23 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C), t, r);

          acc_C30 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C), t, r);
          acc_C31 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C), t, r);
          acc_C32 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C), t, r);
          acc_C33 = aie::transpose(
              aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C), t, r);
        }

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

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            A1 = aie::load_v<MMUL::size_A>(pA2);
            pA2 += MMUL::size_A;
            A2 = aie::load_v<MMUL::size_A>(pA3);
            pA3 += MMUL::size_A;
            A3 = aie::load_v<MMUL::size_A>(pA4);
            pA4 += MMUL::size_A;

            if constexpr (b_row_maj) {
              B0 = aie::load_v<MMUL::size_B>(pB1);
              pB1 += MMUL::size_B * colB;
              B1 = aie::load_v<MMUL::size_B>(pB2);
              pB2 += MMUL::size_B * colB;
              B2 = aie::load_v<MMUL::size_B>(pB3);
              pB3 += MMUL::size_B * colB;
              B3 = aie::load_v<MMUL::size_B>(pB4);
              pB4 += MMUL::size_B * colB;
            } else {
              B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
              pB1 += MMUL::size_B;
              B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
              pB2 += MMUL::size_B;
              B2 = aie::transpose(aie::load_v<MMUL::size_B>(pB3), t, s);
              pB3 += MMUL::size_B;
              B3 = aie::transpose(aie::load_v<MMUL::size_B>(pB4), t, s);
              pB4 += MMUL::size_B;
            }

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

        if constexpr (c_row_maj) {
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
        } else {
          aie::store_v(pC1,
                       aie::transpose(C00.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C01.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC3,
                       aie::transpose(C02.template to_vector<T_out>(), r, t));
          pC3 += MMUL::size_C;
          aie::store_v(pC4,
                       aie::transpose(C03.template to_vector<T_out>(), r, t));
          pC4 += MMUL::size_C;

          aie::store_v(pC1,
                       aie::transpose(C10.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C11.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC3,
                       aie::transpose(C12.template to_vector<T_out>(), r, t));
          pC3 += MMUL::size_C;
          aie::store_v(pC4,
                       aie::transpose(C13.template to_vector<T_out>(), r, t));
          pC4 += MMUL::size_C;

          aie::store_v(pC1,
                       aie::transpose(C20.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C21.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC3,
                       aie::transpose(C22.template to_vector<T_out>(), r, t));
          pC3 += MMUL::size_C;
          aie::store_v(pC4,
                       aie::transpose(C23.template to_vector<T_out>(), r, t));
          pC4 += MMUL::size_C;

          aie::store_v(pC1,
                       aie::transpose(C30.template to_vector<T_out>(), r, t));
          pC1 += MMUL::size_C;
          aie::store_v(pC2,
                       aie::transpose(C31.template to_vector<T_out>(), r, t));
          pC2 += MMUL::size_C;
          aie::store_v(pC3,
                       aie::transpose(C32.template to_vector<T_out>(), r, t));
          pC3 += MMUL::size_C;
          aie::store_v(pC4,
                       aie::transpose(C33.template to_vector<T_out>(), r, t));
          pC4 += MMUL::size_C;
        }
      }
  }

  event1();
}

#ifdef B_COL_MAJ
constexpr bool is_b_row_maj = false;
#else
constexpr bool is_b_row_maj = true;
#endif

#ifdef C_COL_MAJ
constexpr bool is_c_row_maj = false;
#else
constexpr bool is_c_row_maj = true;
#endif

// The following kernel definitions use mmul shapes and kernel expansions that
// have been found to be optimal for AIE2.
//
// All available matrix multiplication shapes in the AIE-API can be found here:
// https://xilinx.github.io/aie_api/group__group__mmul.html
//
// They are all defined based on the shape of the mmul, the input data format
// and the output data format.
//
// Additionally, they check for the correct
// divisibility of the tile dimensions. Note that while both the 'm' and 'n'
// dimensions of the mmul are expanded, the 'k' dimension is not.

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x4_i16_i16(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 4;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<int16, int16, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                      pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x4_i16_i32(const int16 *__restrict pA,
                                                   const int16 *__restrict pB,
                                                   int32 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 4;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_2x2_mmul<int16, int32, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj, is_c_row_maj>(pA, pB,
                                                                      pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_bf16(const bfloat16 *__restrict pA,
                                  const bfloat16 *__restrict pB,
                                  bfloat16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_4x4<bfloat16, bfloat16, (m / r), (k / s), (n / t), r,
                               s, t, is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x8x4_bf16_f32(const bfloat16 *__restrict pA,
                                 const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 4;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (4 * t) == 0);

  return matmul_vectorized_4x4<bfloat16, float, (m / r), (k / s), (n / t), r, s,
                               t, is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_i8_i8(const int8 *__restrict pA,
                                                 const int8 *__restrict pB,
                                                 int8 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_4x2_mmul<int8, int8, (m / r), (k / s), (n / t), r, s,
                                    t, is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_i8_i16(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int16 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_4x2_mmul<int8, int16, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj>(pA, pB, pC);
}

template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_i8_i32(const int8 *__restrict pA,
                                                  const int8 *__restrict pB,
                                                  int32 *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;

  static_assert(m % (4 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  return matmul_vectorized_4x2_mmul<int8, int32, (m / r), (k / s), (n / t), r,
                                    s, t, is_b_row_maj>(pA, pB, pC);
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

#ifdef i8_i8_ONLY
#define combos(X) X(int8, i8, int8, i8, 4, 8, 8)
#endif

#ifdef i8_i16_ONLY
#define combos(X) X(int8, i8, int16, i16, 4, 8, 8)
#endif

#ifdef i8_i32_ONLY
#define combos(X) X(int8, i8, int32, i32, 4, 8, 8)
#endif

#ifdef i16_i16_ONLY
#define combos(X) X(int16, i16, int16, i16, 4, 4, 4)
#endif

#ifdef i16_i32_ONLY
#define combos(X) X(int16, i16, int32, i32, 4, 4, 4)
#endif

#ifdef bf16_bf16_ONLY
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)
#endif

#ifdef bf16_f32_ONLY
#define combos(X) X(bfloat16, bf16, float, f32, 4, 8, 4)
#endif

#ifndef combos
#define combos(X)                                                              \
  X(int8, i8, int8, i8, 4, 8, 8)                                               \
  X(int16, i16, int16, i16, 4, 4, 4)                                           \
  X(int16, i16, int32, i32, 4, 4, 4)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 4, 8, 4)                                   \
  X(bfloat16, bf16, float, f32, 4, 8, 4)
#endif

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
    matmul_scalar<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N, is_b_row_maj,      \
                  is_c_row_maj>(a_in, b_in, c_out);                            \
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

#ifndef SCALAR_ONLY
combos(matmul_vectorized_c_func) combos(zero_vectorized_c_func)
#endif
#ifndef VECTORIZED_ONLY
    combos(matmul_scalar_c_func) combos(zero_scalar_c_func)
#endif

} // extern "C"
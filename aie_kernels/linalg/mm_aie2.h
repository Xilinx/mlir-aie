//===- mm_aie2.h ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

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
          bool b_row_maj = true, bool c_row_maj = true, bool unroll_k = false>
static inline void matmul_vectorized_2x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  // Outer-loop body factored into a lambda so the same code can be wrapped
  // by three differently-parameterised loops below. Per-instantiation iter
  // count must match the actual count: the clang loop pragma takes a literal
  // and silently misbehaves on template-dependent expressions.
  auto outer_body = [&](unsigned z) [[gnu::always_inline]] {
    T_out *__restrict pC1;
    T_out *__restrict pC2;
    if constexpr (c_row_maj) {
      pC1 = pC + (z * colB) * MMUL::size_C;
      pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
    }

    // A cursor that walks the K steps of this row block and wraps back to its
    // start. The address generator keeps the wrap opaque to LICM, so an
    // unrolled K reduction reloads A per 'j' instead of hoisting all of it and
    // spilling.
    const T_in *pAk = pA + (z * colA) * MMUL::size_A;
    dims_2d_t a_dims = dims_2d_from_steps(colA, MMUL::size_A * sizeof(T_in), 0);

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {

        if constexpr (!c_row_maj) {
          pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
          pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
        }
        const T_in *__restrict pBk =
            pB + (b_row_maj ? j : j * colA) * MMUL::size_B;
        constexpr unsigned a_row = colA * MMUL::size_A;
        constexpr unsigned b_tile =
            b_row_maj ? MMUL::size_B : colA * MMUL::size_B;

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

        auto k_step = [&]() {
          A0 = aie::load_v<MMUL::size_A>(pAk);
          A1 = aie::load_v<MMUL::size_A>(pAk + a_row);
          pAk = add_2d_byte(pAk, a_dims);
          if constexpr (b_row_maj) {
            B0 = aie::load_v<MMUL::size_B>(pBk);
            B1 = aie::load_v<MMUL::size_B>(pBk + b_tile);
            pBk += MMUL::size_B * colB;
          } else {
            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pBk), t, s);
            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pBk + b_tile), t, s);
            pBk += MMUL::size_B;
          }

          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);
        };

        // Unrolling K lets 'j' pipeline; see the same switch in mm_aie2p.h.
        if constexpr (unroll_k) {
          AIE_LOOP_UNROLL_FULL
          for (unsigned i = 0; i < colA; ++i)
            k_step();
        } else {
          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
          k_step();
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
  };

  constexpr unsigned outer_iters = rowA / 2;
  if constexpr (outer_iters >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned z = 0; z < rowA; z += 2)
      outer_body(z);
  } else if constexpr (outer_iters >= 2) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (unsigned z = 0; z < rowA; z += 2)
      outer_body(z);
  } else {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 2)
      outer_body(z);
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
          bool b_row_maj = true, bool c_row_maj = true, bool unroll_k = false>
static inline void matmul_vectorized_4x2_mmul(const T_in *__restrict pA,
                                              const T_in *__restrict pB,
                                              T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  auto outer_body = [&](unsigned z) [[gnu::always_inline]] {
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

    // The A cursor and K unroll are as in matmul_vectorized_2x2_mmul.
    const T_in *pAk = pA + (z * colA) * MMUL::size_A;
    dims_2d_t a_dims = dims_2d_from_steps(colA, MMUL::size_A * sizeof(T_in), 0);

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        if constexpr (!c_row_maj) {
          pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
          pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
        }

        const T_in *__restrict pBk =
            pB + (b_row_maj ? j : j * colA) * MMUL::size_B;
        constexpr unsigned a_row = colA * MMUL::size_A;
        constexpr unsigned b_tile =
            b_row_maj ? MMUL::size_B : colA * MMUL::size_B;

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

        auto k_step = [&]() {
          A01 = aie::load_v<MMUL::size_A>(pAk);
          A11 = aie::load_v<MMUL::size_A>(pAk + a_row);
          A21 = aie::load_v<MMUL::size_A>(pAk + 2 * a_row);
          A31 = aie::load_v<MMUL::size_A>(pAk + 3 * a_row);
          pAk = add_2d_byte(pAk, a_dims);
          if constexpr (b_row_maj) {
            B0 = aie::load_v<MMUL::size_B>(pBk);
            B1 = aie::load_v<MMUL::size_B>(pBk + b_tile);
            pBk += MMUL::size_B * colB;
          } else {
            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pBk), t, s);
            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pBk + b_tile), t, s);
            pBk += MMUL::size_B;
          }

          C00.mac(A01, B0);
          C01.mac(A01, B1);
          C10.mac(A11, B0);
          C11.mac(A11, B1);
          C20.mac(A21, B0);
          C21.mac(A21, B1);
          C30.mac(A31, B0);
          C31.mac(A31, B1);
        };

        if constexpr (unroll_k) {
          AIE_LOOP_UNROLL_FULL
          for (unsigned i = 0; i < colA; i += 1)
            k_step();
        } else {
          for (unsigned i = 0; i < colA; i += 1)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
          k_step();
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
  };

  constexpr unsigned outer_iters = rowA / 4;
  if constexpr (outer_iters >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else if constexpr (outer_iters >= 2) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
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
          bool b_row_maj = true, bool c_row_maj = true, bool unroll_k = false>
static inline void matmul_vectorized_4x4(const T_in *__restrict pA,
                                         const T_in *__restrict pB,
                                         T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  auto outer_body = [&](unsigned z) [[gnu::always_inline]] {
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

    // The A cursor and K unroll are as in matmul_vectorized_2x2_mmul.
    const T_in *pAk = pA + (z * colA) * MMUL::size_A;
    dims_2d_t a_dims = dims_2d_from_steps(colA, MMUL::size_A * sizeof(T_in), 0);

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
        const T_in *__restrict pBk =
            pB + (b_row_maj ? j : j * colA) * MMUL::size_B;

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

        constexpr unsigned a_row = colA * MMUL::size_A;
        constexpr unsigned b_tile =
            b_row_maj ? MMUL::size_B : colA * MMUL::size_B;
        auto k_step = [&]() {
          A0 = aie::load_v<MMUL::size_A>(pAk);
          A1 = aie::load_v<MMUL::size_A>(pAk + a_row);
          A2 = aie::load_v<MMUL::size_A>(pAk + 2 * a_row);
          A3 = aie::load_v<MMUL::size_A>(pAk + 3 * a_row);
          pAk = add_2d_byte(pAk, a_dims);

          if constexpr (b_row_maj) {
            B0 = aie::load_v<MMUL::size_B>(pBk);
            B1 = aie::load_v<MMUL::size_B>(pBk + b_tile);
            B2 = aie::load_v<MMUL::size_B>(pBk + 2 * b_tile);
            B3 = aie::load_v<MMUL::size_B>(pBk + 3 * b_tile);
            pBk += MMUL::size_B * colB;
          } else {
            B0 = aie::transpose(aie::load_v<MMUL::size_B>(pBk), t, s);
            B1 = aie::transpose(aie::load_v<MMUL::size_B>(pBk + b_tile), t, s);
            B2 = aie::transpose(aie::load_v<MMUL::size_B>(pBk + 2 * b_tile), t,
                                s);
            B3 = aie::transpose(aie::load_v<MMUL::size_B>(pBk + 3 * b_tile), t,
                                s);
            pBk += MMUL::size_B;
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
        };

        if constexpr (unroll_k) {
          AIE_LOOP_UNROLL_FULL
          for (unsigned i = 0; i < colA; ++i)
            k_step();
        } else {
          for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
            AIE_LOOP_FLATTEN
#endif
          k_step();
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
  };

  constexpr unsigned outer_iters = rowA / 4;
  if constexpr (outer_iters >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else if constexpr (outer_iters >= 2) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
  } else {
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (unsigned z = 0; z < rowA; z += 4)
      outer_body(z);
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

  // Four row blocks per 'j' halve the C loads, conversions and stores per mac
  // against two.
  if constexpr (m % (4 * r) == 0)
    return matmul_vectorized_4x2_mmul<int16, int32, (m / r), (k / s), (n / t),
                                      r, s, t, is_b_row_maj, is_c_row_maj,
                                      (k / s) <= 16>(pA, pB, pC);
  else
    return matmul_vectorized_2x2_mmul<int16, int32, (m / r), (k / s), (n / t),
                                      r, s, t, is_b_row_maj, is_c_row_maj>(
        pA, pB, pC);
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

  // The core powers up in rounding_mode::floor, which biases every bf16
  // conversion toward negative infinity, so the error accumulates over the K
  // reduction rather than cancelling. Under ROUND_CONV_EVEN, select
  // round-to-nearest-even for the duration and restore the caller's mode.
#ifdef ROUND_CONV_EVEN
  aie::rounding_mode saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
#endif
  matmul_vectorized_4x4<bfloat16, bfloat16, (m / r), (k / s), (n / t), r, s, t,
                        is_b_row_maj, is_c_row_maj, (k / s) <= 8>(pA, pB, pC);
#ifdef ROUND_CONV_EVEN
  aie::set_rounding(saved_rounding);
#endif
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

  // See matmul_vectorized_4x8x4_bf16_bf16: the A and B conversions bias the
  // same way whether the accumulator is bf16 or f32.
#ifdef ROUND_CONV_EVEN
  aie::rounding_mode saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
#endif
  matmul_vectorized_4x4<bfloat16, float, (m / r), (k / s), (n / t), r, s, t,
                        is_b_row_maj, is_c_row_maj, (k / s) <= 8>(pA, pB, pC);
#ifdef ROUND_CONV_EVEN
  aie::set_rounding(saved_rounding);
#endif
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
                                    t, is_b_row_maj, is_c_row_maj,
                                    (k / s) <= 4>(pA, pB, pC);
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
                                    s, t, is_b_row_maj, is_c_row_maj,
                                    (k / s) <= 4>(pA, pB, pC);
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
                                    s, t, is_b_row_maj, is_c_row_maj,
                                    (k / s) <= 4>(pA, pB, pC);
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

#ifndef SCALAR_ONLY
combos(matmul_vectorized_c_func)
#endif
#ifndef VECTORIZED_ONLY
    combos(matmul_scalar_c_func)
#endif

} // extern "C"
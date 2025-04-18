//===- mm_b_col_maj.cc ------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

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
 *
 * For matrix A and C, data within each tile (rxs and rxt) are assumed to
 * be in row-major order. Also, the entire tiles themselves are stored in
 * row-major order, as shown in the example below for matrix A:
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
 *
 * However, for matrix B, the data within each tile (sxt) are
 * assumed to be in col-major order. Also, the entire tiles themselves are
 * stored in col-major order, as shown below:
 *
 *      <-t->
 *    _  ________________________
 * 	  s |  1 |  y | ...
 * 	  _ |____|____|
 * 	    |  2 | y+1| ...
 * 	    |____|____|
 *      |  3 | y+2| ...
 *      |____|____|
 * 	    | .    .
 * 	    | .    .
 * 	    | .    .
 */

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void
matmul_vectorized_2x2_mmul_b_col_maj(const T_in *__restrict pA,
                                     const T_in *__restrict pB,
                                     T_out *__restrict pC) {

  using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

  event0();

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      T_out *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
      T_out *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const T_in *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
          const T_in *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
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

          // Load partial results from C buffer for accumulation in-place. The
          // zero.cc function handles the zeroing of data when a new
          // accumulation is needed (after the 'K' reduction dimension)
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
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
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

          // TODO make shift right here to keep most significat bits
          // when lowering the output
          // example below shows how to shift right 10 bits
          // #define SHIFT 10
          // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));
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

// int16 MatMul kernel definion with int16 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x4x8_i16_i16_b_col_maj(const int16 *__restrict pA,
                                          const int16 *__restrict pB,
                                          int16 *__restrict pC) {

  // After extensive experimentation, the 4x4x8 aie::mmul size was found to be
  // optimal for AIE2P, in combination with the 2x2 mmul expanded kernel
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 8;

  // Since the kernel has been expanded twice for both A ('m' dimension) and B
  // ('n' dimension), the following assertions veirify this even division for
  // the single AIE MatMul dimensionality. Notice that 'k' dimension is not
  // spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul_b_col_maj<int16, int16, (m / r), (k / s),
                                              (n / t), r, s, t>(pA, pB, pC);
}

// int16 MatMul kernel definion with int32 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_4x4x8_i16_i32_b_col_maj(const int16 *__restrict pA,
                                          const int16 *__restrict pB,
                                          int32 *__restrict pC) {

  // After extensive experimentation, the 4x4x8 aie::mmul size was found to be
  // optimal for AIE2P, in combination with the 2x2 mmul expanded kernel
  constexpr int r = 4;
  constexpr int s = 4;
  constexpr int t = 8;

  // Since the kernel has been expanded twice for both A ('m' dimension) and B
  // ('n' dimension), the following assertions veirify this even division for
  // the single AIE MatMul dimensionality Notice that 'k' dimension is not
  // spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul_b_col_maj<int16, int32, (m / r), (k / s),
                                              (n / t), r, s, t>(pA, pB, pC);
}

// bf16 MatMul kernel definion with bf16 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_bf16_b_col_maj(const bfloat16 *__restrict pA,
                                            const bfloat16 *__restrict pB,
                                            bfloat16 *__restrict pC) {

  // After extensive experimentation, the 8x8x8 aie::mmul size was found to be
  // optimal for AIE2P, in combination with the 2x2 mmul expanded kernel
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // Since the kernel has been expanded 2 times for both A ('m' dimension) and B
  // ('n' dimension), the following assertions veirify this even division for
  // the single AIE MatMul dimensionality Notice that 'k' dimension is not
  // spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul_b_col_maj<bfloat16, bfloat16, (m / r),
                                              (k / s), (n / t), r, s, t>(pA, pB,
                                                                         pC);
}

// bf16 MatMul kernel definion with fp32 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_bf16_f32_b_col_maj(const bfloat16 *__restrict pA,
                                           const bfloat16 *__restrict pB,
                                           float *__restrict pC) {

  // After extensive experimentation, the 4x8x4 aie::mmul size was found to be
  // optimal for AIE2P, in combination with the 2x2 mmul expanded kernel
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // Since the kernel has been expanded 2 times for both A ('m' dimension) and B
  // ('n' dimension), the following assertions veirify this even division for
  // the single AIE MatMul dimensionality Notice that 'k' dimension is not
  // spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul_b_col_maj<bfloat16, float, (m / r), (k / s),
                                              (n / t), r, s, t>(pA, pB, pC);
}

// int8 MatMul kernel definion with int8 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i8_b_col_maj(
    const int8 *__restrict pA, const int8 *__restrict pB, int8 *__restrict pC) {

  // After extensive experimentation, the 8x8x8 aie::mmul size was found to be
  // optimal for AIE2P, in combination with the 2x2 mmul expanded kernel
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // Since the kernel has been expanded 2 times for A ('m' dimension) and 2
  // times for B ('n' dimension), the following assertions veirify this even
  // division for the single AIE MatMul dimensionality Notice that 'k' dimension
  // is not spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul_b_col_maj<int8, int8, (m / r), (k / s),
                                              (n / t), r, s, t>(pA, pB, pC);
}

// int8 MatMul kernel definion with int16 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_i8_i16_b_col_maj(const int8 *__restrict pA,
                                         const int8 *__restrict pB,
                                         int16 *__restrict pC) {

  // After extensive experimentation, the 8x8x8 aie::mmul size was found to be
  // optimal for AIE2P, in combination with the 2x2 mmul expanded kernel
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // Since the kernel has been expanded 2 times for A ('m' dimension) and 2
  // times for B ('n' dimension), the following assertions veirify this even
  // division for the single AIE MatMul dimensionality Notice that 'k' dimension
  // is not spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul_b_col_maj<int8, int16, (m / r), (k / s),
                                              (n / t), r, s, t>(pA, pB, pC);
}

// int8 MatMul kernel definion with int32 outputs.
template <unsigned m, unsigned k, unsigned n>
static inline void
matmul_vectorized_8x8x8_i8_i32_b_col_maj(const int8 *__restrict pA,
                                         const int8 *__restrict pB,
                                         int32 *__restrict pC) {

  // Since the kernel has been expanded 2 times for A ('m' dimension) and 2
  // times for B ('n' dimension), in combination with the 2x2 mmul expanded
  // kernel
  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  // Since the kernel has been expanded twice for both A ('m' dimension) and B
  // ('n' dimension), the following assertions veirify this even division for
  // the single AIE MatMul dimensionality Notice that 'k' dimension is not
  // spatially expanded.
  static_assert(m % (2 * r) == 0); // 'm' dimension
  static_assert(k % s == 0);       // 'k' dimension
  static_assert(n % (2 * t) == 0); // 'n' dimension

  return matmul_vectorized_2x2_mmul_b_col_maj<int8, int32, (m / r), (k / s),
                                              (n / t), r, s, t>(pA, pB, pC);
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

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
#define combos(X) X(int8, i8, int8, i8, 8, 8, 8)
#endif

#ifdef i8_i16_ONLY
#define combos(X) X(int8, i8, int16, i16, 8, 8, 8)
#endif

#ifdef i8_i32_ONLY
#define combos(X) X(int8, i8, int32, i32, 8, 8, 8)
#endif

#ifdef i16_i16_ONLY
#define combos(X) X(int16, i16, int16, i16, 4, 4, 8)
#endif

#ifdef i16_i32_ONLY
#define combos(X) X(int16, i16, int32, i32, 4, 4, 8)
#endif

#ifdef bf16_bf16_ONLY
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 8, 8, 8)
#endif

#ifdef bf16_f32_ONLY
#define combos(X) X(bfloat16, bf16, float, f32, 8, 8, 8)
#endif

#ifndef combos
#define combos(X)                                                              \
  X(int8, i8, int8, i8, 8, 8, 8)                                               \
  X(int16, i16, int16, i16, 4, 4, 8)                                           \
  X(int16, i16, int32, i32, 4, 4, 8)                                           \
  X(bfloat16, bf16, bfloat16, bf16, 8, 8, 8)                                   \
  X(bfloat16, bf16, float, f32, 8, 8, 8)
#endif

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
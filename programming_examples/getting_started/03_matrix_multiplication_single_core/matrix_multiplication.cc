//===- matrix_multiplication.cc ----------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

#if defined(__chess__)
#define AIE_PREPARE_FOR_PIPELINING [[chess::prepare_for_pipelining]]
#define AIE_LOOP_MIN_ITERATION_COUNT(x) [[chess::min_loop_count(x)]]
#elif defined(__AIECC__)
#ifndef __STRINGIFY
#define __STRINGIFY(a) #a
#endif
#define AIE_LOOP_MIN_ITERATION_COUNT(x)                                        \
  _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_PREPARE_FOR_PIPELINING
#else
#define AIE_LOOP_MIN_ITERATION_COUNT(x)
#define AIE_PREPARE_FOR_PIPELINING
#endif

// Make sure the following tile and intrinsic sizes match the sizes in the
// data layout transformations described in
// matrix_multiplication_single_core.py.
constexpr unsigned m = 64;
constexpr unsigned k = 64;
constexpr unsigned n = 64;
constexpr unsigned r = 8;
constexpr unsigned s = 2;
constexpr unsigned t = 8;

using MMUL = aie::mmul<r, s, t, int16, int16>;

extern "C" {

//

// Multiply A and B, and add the result onto the values already in C.
// A, B, and C must be tiled into tiles of size r*s, s*t, and r*t,
// respectively (in our design, the DMA performs this tiling).
void matrix_multiplication(const int16 *__restrict A, const int16 *__restrict B,
                           int16 *__restrict C) {
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned row = 0; row < m / r; row += 2) {
    for (unsigned col = 0; col < n / t; col += 2) {

      // The following pointers point to the start of two rows of A and
      // tow columns of B, respectively.
      const int16 *__restrict A0_ptr =
          A + ((row + 0) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict A1_ptr =
          A + ((row + 1) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict B0_ptr =
          B + (0 * (n / t) + (col + 0)) * MMUL::size_B;
      const int16 *__restrict B1_ptr =
          B + (0 * (n / t) + (col + 1)) * MMUL::size_B;

      const aie::vector<int16, MMUL::size_C> C00_in = aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C);
      const aie::vector<int16, MMUL::size_C> C01_in = aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C);
      const aie::vector<int16, MMUL::size_C> C10_in = aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C);
      const aie::vector<int16, MMUL::size_C> C11_in = aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C);
      MMUL C00(C00_in);
      MMUL C01(C01_in);
      MMUL C10(C10_in);
      MMUL C11(C11_in);

      // The following loop iterates over the k dimension of the
      // input matrices, i.e., over each tile in the same row of
      // A and each tile in the same column of B.
      for (unsigned i = 0; i < k / s; i += 1, A0_ptr += MMUL::size_A,
                    A1_ptr += MMUL::size_A, B0_ptr += (n / t) * MMUL::size_B,
                    B1_ptr += (n / t) * MMUL::size_B) {
        const aie::vector<int16, MMUL::size_A> A0 =
            aie::load_v<MMUL::size_A>(A0_ptr);
        const aie::vector<int16, MMUL::size_A> A1 =
            aie::load_v<MMUL::size_A>(A1_ptr);
        const aie::vector<int16, MMUL::size_B> B0 =
            aie::load_v<MMUL::size_B>(B0_ptr);
        const aie::vector<int16, MMUL::size_B> B1 =
            aie::load_v<MMUL::size_B>(B1_ptr);
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
      aie::store_v(C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C,
                   C00.template to_vector<int16>());
      aie::store_v(C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C,
                   C01.template to_vector<int16>());
      aie::store_v(C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C,
                   C10.template to_vector<int16>());
      aie::store_v(C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C,
                   C11.template to_vector<int16>());
    }
  }
}

// The following function is not used in our design, but may help to
// visualize the data access pattern of the above vectorized version.
void matrix_multiplication_scalar(const int16 *__restrict A,
                                  const int16 *__restrict B,
                                  int16 *__restrict C) {
  for (unsigned tile_row = 0; tile_row < m / r; tile_row += 1) {
    for (unsigned tile_col = 0; tile_col < n / t; tile_col += 1) {
      for (unsigned tile_i = 0; tile_i < k / s; tile_i += 1) {
        const int16 *A_base = A + (tile_row * (k / s) + tile_i) * r * s;
        const int16 *B_base = B + (tile_i * (n / t) + tile_col) * s * t;
        int16 *C_base = C + (tile_row * (n / t) + tile_col) * r * t;
        for (unsigned row = 0; row < r; row += 1) {
          for (unsigned col = 0; col < t; col += 1) {
            for (unsigned i = 0; i < s; i += 1) {
              C_base[row * t + col] +=
                  A_base[row * s + i] * B_base[i * t + col];
            }
          }
        }
      }
    }
  }
}
}

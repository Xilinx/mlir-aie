//===- mv.cc ----------------------------------------------000---*- C++ -*-===//
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

template <typename T_in, typename T_out, int M, int K>
void matvec_scalar(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < M; row++) {
    T_out runningSum = 0;
    for (int i = 0; i < K; i++) {
      runningSum += a[row * K + i] * b[i];
    }
    c[row] += runningSum;
  }
  event1();
}

template <typename T_in, typename T_out, typename T_acc, unsigned m, unsigned k,
          unsigned r, unsigned s>
void matvec_vectorized(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(m % r == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 because that is the number of
                         // column vectors (a_vec_0_0..a_vec_3_1) we create
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value);

  // This kernel expects a "32-bit word transposed matrix", i.e. the result
  // of transposing the row-major representation of the matrix at a
  // granularity of 4 bytes. For the bf16 data type of the inputs, this
  // corresponds to a memory layout like this:
  //  1  2  9 10 17 18
  //  3  4 11 12 19 ..
  //  5  6 13 14
  //  7  8 15 16

  // In the outer loop, we iterate through the b matrix once, in steps of
  // 8*1-sized blocks.
  //
  // In the inner loop, we iterate through blocks of the A matrix in
  // colum-major order, at each step consuming a r*8-sized block.
  //
  // At each iteration, we accumulate into r rows of the output. To
  // accumulate, we add the dot product of each row of A with the same
  // acquired b vector from the outer loop.

  event0();
  T_in *__restrict a_ptr = a;
  T_in *__restrict b_ptr = b;

  for (int col = 0; col < k; col += 8) {
    aie::vector<T_in, 8> b_vec = aie::load_v<8>(b_ptr);
    T_out *__restrict c_ptr = c; // reset to the first row of C output on
                                 // each outer loop tieration

    for (int row = 0; row < m; row += r)
      chess_loop_range(m / r, ) {
        aie::accum<T_acc, r> c_acc_in;
        c_acc_in.from_vector(aie::load_v<r>(c_ptr));

        const aie::vector<T_in, 2 *r> a_vec_0 = aie::load_v<2 * r>(a_ptr);
        const aie::vector<T_in, 2 *r> a_vec_1 =
            aie::load_v<2 * r>(a_ptr + 2 * m);
        const aie::vector<T_in, 2 *r> a_vec_2 =
            aie::load_v<2 * r>(a_ptr + 4 * m);
        const aie::vector<T_in, 2 *r> a_vec_3 =
            aie::load_v<2 * r>(a_ptr + 6 * m);

        // The even/odd calls below extract the interleaved columns of A.
        // We need to do this since A is only transposed (column-major) at
        // a granularity of 4 bytes, but bf16 are two bytes; therefore, we
        // end up with two interleaved columns at each 2*m interval.
        // After this, each of a_vec_0_0 contains rows row..row+r of some
        // column of A. The columns are col..col+8.
        const aie::vector<T_in, r> a_vec_0_0 = aie::filter_even(a_vec_0);
        const aie::vector<T_in, r> a_vec_0_1 = aie::filter_odd(a_vec_0);
        const aie::vector<T_in, r> a_vec_1_0 = aie::filter_even(a_vec_1);
        const aie::vector<T_in, r> a_vec_1_1 = aie::filter_odd(a_vec_1);
        const aie::vector<T_in, r> a_vec_2_0 = aie::filter_even(a_vec_2);
        const aie::vector<T_in, r> a_vec_2_1 = aie::filter_odd(a_vec_2);
        const aie::vector<T_in, r> a_vec_3_0 = aie::filter_even(a_vec_3);
        const aie::vector<T_in, r> a_vec_3_1 = aie::filter_odd(a_vec_3);

        // The accumulate call below produces the following output:
        // c_acc_out[i] = c_acc_in + b_vec[0]*a_vec_0_0[i]
        //                         + b_vec[1]*a_vec_0_1[i]
        //                         + ...
        //                         + b_vec[7]*a_vec_3_1[i]
        // i.e., the dot product of vector b_vec with one row (row+i)
        // (recall that the different a_vecs are columns, thus we are
        // indexing into the same row i for each column).
        // The same could be implemented with a sequence of aie::muls (one
        // aie::mac to add the accumulator c_in), and then aie::adding all
        // the resulting vectors together.
        auto c_acc_out = aie::accumulate<r>(
            c_acc_in, b_vec, 0, a_vec_0_0, a_vec_0_1, a_vec_1_0, a_vec_1_1,
            a_vec_2_0, a_vec_2_1, a_vec_3_0, a_vec_3_1);

        aie::store_v(c_ptr, c_acc_out.template to_vector<T_out>());
        a_ptr += 2 * r; // On last iteration, this advances to next column.
                        // This is why we only iterate by 6*m in the outer
                        // loop, for a total of 8*m, i.e. 8 columns.
        c_ptr += r;     // Move to next r rows of the same columns in A.
      }

    a_ptr += 6 * m; // Move to next 8 columns of A.
    b_ptr += s;     // Move to next s (==8) rows of b.
  }
  event1();
}

extern "C" {

#define combos(X)                                                              \
  X(bfloat16, bf16, float, f32, accfloat)                                      \
//    X(int16,         i16, int16,    i16, acc32)                                  \

#define matvec_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             ctype_acc)                                        \
  void matvec_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matvec_scalar<ctype_in, ctype_out, 32, 32>(a_in, b_in, c_out);             \
  }

#define matvec_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, ctype_acc)                     \
  void matvec_vectorized_##mlir_type_in##_##mlir_type_out(                     \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matvec_vectorized<ctype_in, ctype_out, ctype_acc, 32, 32, 16, 8>(          \
        a_in, b_in, c_out);                                                    \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out, ctype_acc)                       \
  void zero_vectorized_##mlir_type_out(ctype_out *c_out) {                     \
    zero_vectorized<ctype_out, 32, 1, 32>(c_out);                              \
  }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out,   \
                           ctype_acc)                                          \
  void zero_scalar_##mlir_type_out(ctype_out *c_out) {                         \
    zero_scalar<ctype_out, 32, 1>(c_out);                                      \
  }

combos(matvec_scalar_c_func) combos(matvec_vectorized_c_func)
    combos(zero_vectorized_c_func) combos(zero_scalar_c_func)

} // extern "C"

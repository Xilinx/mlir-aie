//===- mv_i16.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// int16 x int16 -> int32 matrix-vector multiply; the vectorized path reads A
// word-transposed. mv_bf16.cc is the bf16 counterpart, IRON's GEMV, with a
// row-major A and a different signature.

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

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

#if AIE_TUNED_AIE2
// AIE2, int16: a 32-lane load of the word-transposed A holds 16 rows of one
// column pair, row i's even column in lane 2i and its odd column in lane
// 2i + 1. An elementwise mac against that column pair of b, repeated 16
// times, accumulates both columns without separating them; the even and odd
// lanes are added once per row block.
template <unsigned m, unsigned k, unsigned r>
__aie_inline void matvec_i16_aie2(const int16 *__restrict a,
                                  const int16 *__restrict b,
                                  int32 *__restrict c) {
  using acc_t = aie::accum<acc32, 2 * r>;
  auto b_pair = [](const aie::vector<int16, 8> &b_vec, unsigned p) {
    return aie::broadcast<int32, r>(b_vec.template cast_to<int32>()[p])
        .template cast_to<int16>();
  };
  auto finish = [](int32 *__restrict c_ptr, const acc_t &acc) {
    const auto v = acc.template to_vector<int32>();
    aie::store_v(c_ptr, aie::add(aie::load_v<r>(c_ptr),
                                 aie::add(aie::filter_even(v, 1),
                                          aie::filter_odd(v, 1))));
  };

  event0();
  unsigned row = 0;
  for (; row + 2 * r <= m; row += 2 * r) {
    const int16 *__restrict a_lo = a + 2 * row;
    const int16 *__restrict a_hi = a + 2 * row + 2 * r;
    const int16 *__restrict b_ptr = b;
    acc_t acc_lo[2] = {aie::zeros<acc32, 2 * r>(), aie::zeros<acc32, 2 * r>()};
    acc_t acc_hi[2] = {aie::zeros<acc32, 2 * r>(), aie::zeros<acc32, 2 * r>()};

    AIE_LOOP_MIN_ITERATION_COUNT(k / 8)
    AIE_LOOP_UNROLL(k / 8 <= 4 ? k / 8 : 1)
    for (unsigned col = 0; col < k; col += 8) {
      const aie::vector<int16, 8> b_vec = aie::load_v<8>(b_ptr);
      AIE_LOOP_UNROLL_FULL
      for (unsigned p = 0; p < 4; p++) {
        const auto bp = b_pair(b_vec, p);
        acc_lo[p % 2] =
            aie::mac(acc_lo[p % 2], aie::load_v<2 * r>(a_lo + p * 2 * m), bp);
        acc_hi[p % 2] =
            aie::mac(acc_hi[p % 2], aie::load_v<2 * r>(a_hi + p * 2 * m), bp);
      }
      a_lo += 8 * m;
      a_hi += 8 * m;
      b_ptr += 8;
    }
    finish(c + row, aie::add(acc_lo[0], acc_lo[1]));
    finish(c + row + r, aie::add(acc_hi[0], acc_hi[1]));
  }

  if constexpr ((m / r) % 2 != 0) {
    const int16 *__restrict a_ptr = a + 2 * row;
    const int16 *__restrict b_ptr = b;
    acc_t acc = aie::zeros<acc32, 2 * r>();

    AIE_LOOP_MIN_ITERATION_COUNT(k / 8)
    for (unsigned col = 0; col < k; col += 8) {
      const aie::vector<int16, 8> b_vec = aie::load_v<8>(b_ptr);
      AIE_LOOP_UNROLL_FULL
      for (unsigned p = 0; p < 4; p++)
        acc = aie::mac(acc, aie::load_v<2 * r>(a_ptr + p * 2 * m),
                       b_pair(b_vec, p));
      a_ptr += 8 * m;
      b_ptr += 8;
    }
    finish(c + row, acc);
  }
  event1();
}
#endif

template <typename T_in, typename T_out, typename T_acc, unsigned m, unsigned k,
          unsigned r, unsigned s>
void matvec_vectorized(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(m % r == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 because that is the number of
                         // column vectors the four A loads below split into
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value ||
                std::is_same<T_in, int16_t>::value);
#if AIE_TUNED_AIE2
  if constexpr (std::is_same<T_in, int16_t>::value) {
    matvec_i16_aie2<m, k, r>(a, b, c);
    return;
  }
#endif

  // This kernel expects a "32-bit word transposed matrix", i.e. the result
  // of transposing the row-major representation of the matrix at a
  // granularity of 4 bytes. For the bf16 data type of the inputs, this
  // corresponds to a memory layout like this:
  //  1  2  9 10 17 18
  //  3  4 11 12 19 ..
  //  5  6 13 14
  //  7  8 15 16

  // The r*8 block of A holding rows row..row+r of columns col..col+8 starts
  // at a + 8*m*(col/8) + 2*row, with its four loads 2*m apart.
  //
  // The even/odd calls below extract the interleaved columns of A.
  // We need to do this since A is only transposed (column-major) at
  // a granularity of 4 bytes, but bf16 are two bytes; therefore, we
  // end up with two interleaved columns at each 2*m interval.
  // After this, each filtered vector contains rows row..row+r of one
  // column of A. The columns are col..col+8.
  //
  // The accumulate call below produces the following output:
  // acc[i] = acc[i] + b_vec[0]*filter_even(a_vec_0)[i]
  //                 + b_vec[1]*filter_odd(a_vec_0)[i]
  //                 + ...
  //                 + b_vec[7]*filter_odd(a_vec_3)[i]
  // i.e., the dot product of vector b_vec with one row (row+i)
  // (recall that the different a_vecs are columns, thus we are
  // indexing into the same row i for each column).
  // The same could be implemented with a sequence of aie::muls (one
  // aie::mac to add the incoming accumulator), and then aie::adding
  // all the resulting vectors together.
  auto mac_block = [](const T_in *__restrict a_ptr,
                      const aie::vector<T_in, s> &b_vec,
                      aie::accum<T_acc, r> &acc) {
    const aie::vector<T_in, 2 * r> a_vec_0 = aie::load_v<2 * r>(a_ptr);
    const aie::vector<T_in, 2 * r> a_vec_1 = aie::load_v<2 * r>(a_ptr + 2 * m);
    const aie::vector<T_in, 2 * r> a_vec_2 = aie::load_v<2 * r>(a_ptr + 4 * m);
    const aie::vector<T_in, 2 * r> a_vec_3 = aie::load_v<2 * r>(a_ptr + 6 * m);
    acc = aie::accumulate<r>(
        acc, b_vec, 0, aie::filter_even(a_vec_0), aie::filter_odd(a_vec_0),
        aie::filter_even(a_vec_1), aie::filter_odd(a_vec_1),
        aie::filter_even(a_vec_2), aie::filter_odd(a_vec_2),
        aie::filter_even(a_vec_3), aie::filter_odd(a_vec_3));
  };

  event0();

  // Columns are the inner loop and two row blocks share each pass over them,
  // so one C accumulator pair stays in registers for the whole k sweep and one
  // set of b lane broadcasts feeds two mac chains.
  unsigned row = 0;
  for (; row + 2 * r <= m; row += 2 * r) {
    const T_in *__restrict a_ptr = a + 2 * row;
    const T_in *__restrict b_ptr = b;
    aie::accum<T_acc, r> acc_lo, acc_hi;
    acc_lo.from_vector(aie::load_v<r>(c + row));
    acc_hi.from_vector(aie::load_v<r>(c + row + r));

    AIE_LOOP_MIN_ITERATION_COUNT(k / s)
    for (unsigned col = 0; col < k; col += s) {
      const aie::vector<T_in, s> b_vec = aie::load_v<s>(b_ptr);
      mac_block(a_ptr, b_vec, acc_lo);
      mac_block(a_ptr + 2 * r, b_vec, acc_hi);
      a_ptr += s * m; // Move to next 8 columns of A.
      b_ptr += s;     // Move to next s (==8) rows of b.
    }

    aie::store_v(c + row, acc_lo.template to_vector<T_out>());
    aie::store_v(c + row + r, acc_hi.template to_vector<T_out>());
  }

  // m / r need not be even.
  if constexpr ((m / r) % 2 != 0) {
    const T_in *__restrict a_ptr = a + 2 * row;
    const T_in *__restrict b_ptr = b;
    aie::accum<T_acc, r> acc;
    acc.from_vector(aie::load_v<r>(c + row));

    AIE_LOOP_MIN_ITERATION_COUNT(k / s)
    for (unsigned col = 0; col < k; col += s) {
      mac_block(a_ptr, aie::load_v<s>(b_ptr), acc);
      a_ptr += s * m;
      b_ptr += s;
    }

    aie::store_v(c + row, acc.template to_vector<T_out>());
  }
  event1();
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M and DIM_K at compile time using -DDIM_M 16 etc.
// These dimensions must be divisible by the r, s dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 32
#endif

#define combos(X)                                                              \
  /* X(bfloat16, bf16, float, f32, accfloat) */                                \
  X(int16, i16, int32, i32, acc32)

#define matvec_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, \
                             ctype_acc)                                        \
  void matvec_scalar_##mlir_type_in##_##mlir_type_out(                         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matvec_scalar<ctype_in, ctype_out, DIM_M, DIM_K>(a_in, b_in, c_out);       \
  }

#define matvec_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out, ctype_acc)                     \
  void matvec_vectorized_##mlir_type_in##_##mlir_type_out(                     \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    matvec_vectorized<ctype_in, ctype_out, ctype_acc, DIM_M, DIM_K, 16, 8>(    \
        a_in, b_in, c_out);                                                    \
  }

combos(matvec_scalar_c_func) combos(matvec_vectorized_c_func)

} // extern "C"

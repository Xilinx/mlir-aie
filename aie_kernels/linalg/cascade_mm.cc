//===- cascade_mm.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "../aie_kernel_utils.h"

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_only(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = undef_v16int32();
      v16 = upd_elem(v16, 0, (int)running_sum);
      put_mcd(v16);
    }
  }
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_get_only(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += ext_elem(v16, 0U);
      c[row * colB + col] += running_sum;
    }
  }
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_get(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_out running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += a[row * colA + i] * b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += ext_elem(v16, 0U);
      v16 = upd_elem(v16, 0, (int)running_sum);
      put_mcd(v16);
    }
  }
  event1();
}

#if AIE_TUNED_AIE2P
// AIE2P: aie::mmul over the row-major operands, a 2x2 block of C tiles per
// step. Each C tile's accumulator crosses the cascade whole, in the same tile
// order on every half, so a float chain rounds once, at the GET half. Shapes
// that do not tile fall back to the scalar kernels.
template <typename T_in, typename T_out>
struct cascade_tile;

template <typename T_out>
struct cascade_tile<int16, T_out> {
  using MMUL = aie::mmul<4, 4, 8, int16, int16>;
  static constexpr int ks = 8;

  // A rows i..i+3, columns k..k+7: the tiles for k and k + 4.
  static inline void mac_step(MMUL &C00, MMUL &C01, MMUL &C10, MMUL &C11,
                              const int16 *__restrict a0,
                              const int16 *__restrict a1,
                              const int16 *__restrict b, int colA, int colB) {
    auto [a00, a01] = split_a(a0, colA);
    auto [a10, a11] = split_a(a1, colA);
    auto [b00, b01] = split_b(b, colB);
    auto [b10, b11] = split_b(b + 4 * colB, colB);
    C00.mac(a00, b00);
    C01.mac(a00, b01);
    C10.mac(a10, b00);
    C11.mac(a10, b01);
    C00.mac(a01, b10);
    C01.mac(a01, b11);
    C10.mac(a11, b10);
    C11.mac(a11, b11);
  }

  static inline std::pair<aie::vector<int16, 16>, aie::vector<int16, 16>>
  split_a(const int16 *__restrict a, int colA) {
    aie::vector<int16, 32> v =
        aie::concat(aie::load_v<8>(a), aie::load_v<8>(a + colA),
                    aie::load_v<8>(a + 2 * colA), aie::load_v<8>(a + 3 * colA));
    return aie::interleave_unzip(v.extract<16>(0), v.extract<16>(1), 4);
  }

  // B rows k..k+3, columns j..j+15: the tiles for j and j + 8.
  static inline std::pair<aie::vector<int16, 32>, aie::vector<int16, 32>>
  split_b(const int16 *__restrict b, int colB) {
    aie::vector<int16, 64> v = aie::concat(
        aie::load_v<16>(b), aie::load_v<16>(b + colB),
        aie::load_v<16>(b + 2 * colB), aie::load_v<16>(b + 3 * colB));
    return aie::interleave_unzip(v.extract<32>(0), v.extract<32>(1), 8);
  }
};

template <typename T_out>
struct cascade_tile<bfloat16, T_out> {
  using MMUL = aie::mmul<4, 8, 8, bfloat16, bfloat16>;
  static constexpr int ks = 8;

  static inline void mac_step(MMUL &C00, MMUL &C01, MMUL &C10, MMUL &C11,
                              const bfloat16 *__restrict a0,
                              const bfloat16 *__restrict a1,
                              const bfloat16 *__restrict b, int colA,
                              int colB) {
    aie::vector<bfloat16, 32> A0 = load_a(a0, colA);
    aie::vector<bfloat16, 32> A1 = load_a(a1, colA);
    auto [t0, t1] = split_b(b, colB);
    auto [u0, u1] = split_b(b + 4 * colB, colB);
    aie::vector<bfloat16, 64> B0 = aie::concat(t0, u0);
    aie::vector<bfloat16, 64> B1 = aie::concat(t1, u1);
    C00.mac(A0, B0);
    C01.mac(A0, B1);
    C10.mac(A1, B0);
    C11.mac(A1, B1);
  }

  static inline aie::vector<bfloat16, 32> load_a(const bfloat16 *__restrict a,
                                                 int colA) {
    return aie::concat(aie::load_v<8>(a), aie::load_v<8>(a + colA),
                       aie::load_v<8>(a + 2 * colA),
                       aie::load_v<8>(a + 3 * colA));
  }

  static inline std::pair<aie::vector<bfloat16, 32>, aie::vector<bfloat16, 32>>
  split_b(const bfloat16 *__restrict b, int colB) {
    aie::vector<bfloat16, 64> v = aie::concat(
        aie::load_v<16>(b), aie::load_v<16>(b + colB),
        aie::load_v<16>(b + 2 * colB), aie::load_v<16>(b + 3 * colB));
    return aie::interleave_unzip(v.extract<32>(0), v.extract<32>(1), 8);
  }
};

template <typename MMUL>
static inline typename MMUL::accum_type zero_acc() {
  using Acc = typename MMUL::accum_type;
  return aie::zeros<typename Acc::value_type, Acc::size()>();
}

template <typename Acc>
static inline void cascade_put(const Acc &acc) {
  put_mcd(acc.template extract<16>(0));
  put_mcd(acc.template extract<16>(1));
}

template <typename Acc>
static inline Acc cascade_get() {
  using Half = decltype(std::declval<Acc>().template extract<16>(0));
  Acc acc;
  if constexpr (Acc::is_floating_point()) {
    acc.insert(0, Half(get_scd_v16accfloat()));
    acc.insert(1, Half(get_scd_v16accfloat()));
  } else {
    acc.insert(0, Half(get_scd_v16acc64()));
    acc.insert(1, Half(get_scd_v16acc64()));
  }
  return acc;
}

template <typename T_out, typename Acc>
static inline Acc load_c(const T_out *__restrict c, int colB) {
  aie::vector<T_out, 32> v =
      aie::concat(aie::load_v<8>(c), aie::load_v<8>(c + colB),
                  aie::load_v<8>(c + 2 * colB), aie::load_v<8>(c + 3 * colB));
  Acc acc;
  acc.from_vector(v);
  return acc;
}

template <typename T_out, typename Acc>
static inline void store_c(T_out *__restrict c, const Acc &acc, int colB) {
  aie::vector<T_out, 32> v = acc.template to_vector<T_out>();
  AIE_LOOP_UNROLL_FULL
  for (int r = 0; r < 4; r++)
    aie::store_v(c + r * colB, v.template extract<8>(r));
}

template <bool get, bool put, typename T_out, typename MMUL>
static inline void cascade_finish(const MMUL &C, T_out *__restrict c,
                                  int colB) {
  using Acc = typename MMUL::accum_type;
  Acc acc = C.to_accum();
  if constexpr (get)
    acc = aie::add(acc, cascade_get<Acc>());
  if constexpr (put) {
    cascade_put(acc);
  } else {
    acc = aie::add(acc, load_c<T_out, Acc>(c, colB));
    store_c(c, acc, colB);
  }
}

// get: C += own + cascade. put: cascade = own (+ cascade when get is set).
template <bool get, bool put, typename T_in, typename T_out, int rowA, int colA,
          int colB>
static inline void cascade_vector(const T_in *__restrict a,
                                  const T_in *__restrict b,
                                  T_out *__restrict c) {
  using tile = cascade_tile<T_in, T_out>;
  using MMUL = typename tile::MMUL;
  for (int i = 0; i < rowA; i += 8) {
    for (int j = 0; j < colB; j += 16) {
      // Zeroed explicitly: with the first-mac zero flag, Peano drops the
      // first iteration of a pipelined k loop that runs twice.
      MMUL C00(zero_acc<MMUL>()), C01(zero_acc<MMUL>());
      MMUL C10(zero_acc<MMUL>()), C11(zero_acc<MMUL>());
      const T_in *__restrict a0 = a + i * colA;
      const T_in *__restrict a1 = a0 + 4 * colA;
      const T_in *__restrict pb = b + j;
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(colA / tile::ks)
      for (int k = 0; k < colA; k += tile::ks) {
        tile::mac_step(C00, C01, C10, C11, a0 + k, a1 + k, pb, colA, colB);
        pb += tile::ks * colB;
      }
      T_out *pc = c + i * colB + j;
      cascade_finish<get, put, T_out>(C00, pc, colB);
      cascade_finish<get, put, T_out>(C01, pc + 8, colB);
      cascade_finish<get, put, T_out>(C10, pc + 4 * colB, colB);
      cascade_finish<get, put, T_out>(C11, pc + 4 * colB + 8, colB);
    }
  }
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
static constexpr bool cascade_tiles =
    rowA % 8 == 0 && colA % 8 == 0 && colB % 16 == 0;

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_vector_cascade_put_only(T_in *a, T_in *b, T_out *c) {
  if constexpr (!cascade_tiles<T_in, T_out, rowA, colA, colB>) {
    matmul_scalar_cascade_put_only<T_in, T_out, rowA, colA, colB>(a, b, c);
  } else {
    event0();
    cascade_vector<false, true, T_in, T_out, rowA, colA, colB>(a, b, c);
    event1();
  }
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_vector_cascade_put_get(T_in *a, T_in *b, T_out *c) {
  if constexpr (!cascade_tiles<T_in, T_out, rowA, colA, colB>) {
    matmul_scalar_cascade_put_get<T_in, T_out, rowA, colA, colB>(a, b, c);
  } else {
    event0();
    cascade_vector<true, true, T_in, T_out, rowA, colA, colB>(a, b, c);
    event1();
  }
}

// Integer outputs wrap like the scalar kernel's; float outputs round to
// nearest even.
template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_vector_cascade_get_only(T_in *a, T_in *b, T_out *c) {
  if constexpr (!cascade_tiles<T_in, T_out, rowA, colA, colB>) {
    matmul_scalar_cascade_get_only<T_in, T_out, rowA, colA, colB>(a, b, c);
  } else {
    event0();
    aie::saturation_mode saved_saturation =
        aie::swap_saturation(aie::saturation_mode::none);
    aie::rounding_mode saved_rounding =
        aie::swap_rounding(aie::rounding_mode::conv_even);
    cascade_vector<true, false, T_in, T_out, rowA, colA, colB>(a, b, c);
    aie::set_rounding(saved_rounding);
    aie::set_saturation(saved_saturation);
    event1();
  }
}

#define CASCADE_KERNEL(half) matmul_vector_cascade_##half
#else
#define CASCADE_KERNEL(half) matmul_scalar_cascade_##half
#endif

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

#define matmul_scalar_cascade_get_only_c_func(                                 \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_get_only_##mlir_type_in##_##mlir_type_out(        \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    CASCADE_KERNEL(get_only)<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(        \
        a_in, b_in, c_out);                                                    \
  }

#define matmul_scalar_cascade_put_only_c_func(                                 \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_put_only_##mlir_type_in##_##mlir_type_out(        \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    CASCADE_KERNEL(put_only)<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(        \
        a_in, b_in, c_out);                                                    \
  }

#define matmul_scalar_cascade_put_get_c_func(                                  \
    ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)                 \
  void matmul_scalar_cascade_put_get_##mlir_type_in##_##mlir_type_out(         \
      ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {                      \
    CASCADE_KERNEL(put_get)<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N>(         \
        a_in, b_in, c_out);                                                    \
  }

combos(matmul_scalar_cascade_get_only_c_func)
    combos(matmul_scalar_cascade_put_only_c_func)
        combos(matmul_scalar_cascade_put_get_c_func)
} // extern "C"

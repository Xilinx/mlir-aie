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

// A float chain sums in float and sends the float's bits over the cascade.
template <typename T_out>
using cascade_sum_t =
    std::conditional_t<std::is_same_v<T_out, bfloat16>, float, T_out>;

template <typename T>
static inline int to_cascade_word(T sum) {
  if constexpr (std::is_same_v<T, float>)
    return __builtin_bit_cast(int, sum);
  else
    return (int)sum;
}

template <typename T>
static inline T from_cascade_word(int word) {
  if constexpr (std::is_same_v<T, float>)
    return __builtin_bit_cast(float, word);
  else
    return word;
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_only(T_in *a, T_in *b, T_out *c) {
  using T_sum = cascade_sum_t<T_out>;
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_sum running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += (T_sum)a[row * colA + i] * (T_sum)b[i * colB + col];
      }
      v16int32 v16 = undef_v16int32();
      v16 = upd_elem(v16, 0, to_cascade_word(running_sum));
      put_mcd(v16);
    }
  }
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_get_only(T_in *a, T_in *b, T_out *c) {
  using T_sum = cascade_sum_t<T_out>;
  event0();
  aie::rounding_mode saved_rounding = aie::rounding_mode::floor;
  if constexpr (std::is_same_v<T_out, bfloat16>)
    saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_sum running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += (T_sum)a[row * colA + i] * (T_sum)b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += from_cascade_word<T_sum>(ext_elem(v16, 0U));
      c[row * colB + col] += running_sum;
    }
  }
  if constexpr (std::is_same_v<T_out, bfloat16>)
    aie::set_rounding(saved_rounding);
  event1();
}

template <typename T_in, typename T_out, int rowA, int colA, int colB>
void matmul_scalar_cascade_put_get(T_in *a, T_in *b, T_out *c) {
  using T_sum = cascade_sum_t<T_out>;
  event0();
  for (int row = 0; row < rowA; row++) {
    for (int col = 0; col < colB; col++) {
      T_sum running_sum = 0;
      for (int i = 0; i < colA; i++) {
        running_sum += (T_sum)a[row * colA + i] * (T_sum)b[i * colB + col];
      }
      v16int32 v16 = get_scd_v16int32();
      running_sum += from_cascade_word<T_sum>(ext_elem(v16, 0U));
      v16 = upd_elem(v16, 0, to_cascade_word(running_sum));
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
  using A = aie::vector<int16, 16>;
  // Steps of 16 when the A rows allow 256-bit loads.
  template <int colA>
  static constexpr int ks = colA % 16 == 0 ? 16 : 8;

  template <int colA>
  static inline void mac_step(MMUL &C00, MMUL &C01, MMUL &C10, MMUL &C11,
                              const int16 *__restrict a0,
                              const int16 *__restrict a1,
                              const int16 *__restrict b, int colB) {
    if constexpr (ks<colA> == 16) {
      auto [a00, a01, a02, a03] = split_a16(a0, colA);
      auto [a10, a11, a12, a13] = split_a16(a1, colA);
      mac4(C00, C01, C10, C11, a00, a10, b, colB);
      mac4(C00, C01, C10, C11, a01, a11, b + 4 * colB, colB);
      mac4(C00, C01, C10, C11, a02, a12, b + 8 * colB, colB);
      mac4(C00, C01, C10, C11, a03, a13, b + 12 * colB, colB);
    } else {
      auto [a00, a01] = split_a(a0, colA);
      auto [a10, a11] = split_a(a1, colA);
      mac4(C00, C01, C10, C11, a00, a10, b, colB);
      mac4(C00, C01, C10, C11, a01, a11, b + 4 * colB, colB);
    }
  }

  static inline void mac4(MMUL &C00, MMUL &C01, MMUL &C10, MMUL &C11, A a0,
                          A a1, const int16 *__restrict b, int colB) {
    auto [b0, b1] = split_b(b, colB);
    C00.mac(a0, b0);
    C01.mac(a0, b1);
    C10.mac(a1, b0);
    C11.mac(a1, b1);
  }

  // A rows i..i+3, columns k..k+15: the tiles for k, k + 4, k + 8, k + 12.
  static inline std::array<A, 4> split_a16(const int16 *__restrict a,
                                           int colA) {
    auto [e, o] = aie::interleave_unzip(
        aie::concat(aie::load_v<16>(a), aie::load_v<16>(a + colA)),
        aie::concat(aie::load_v<16>(a + 2 * colA),
                    aie::load_v<16>(a + 3 * colA)),
        4);
    auto [t0, t8] =
        aie::interleave_unzip(e.extract<16>(0), e.extract<16>(1), 4);
    auto [t4, t12] =
        aie::interleave_unzip(o.extract<16>(0), o.extract<16>(1), 4);
    return {t0, t4, t8, t12};
  }

  // A rows i..i+3, columns k..k+7: the tiles for k and k + 4.
  static inline std::pair<A, A> split_a(const int16 *__restrict a, int colA) {
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
  template <int colA>
  static constexpr int ks = 8;

  template <int colA>
  static inline void mac_step(MMUL &C00, MMUL &C01, MMUL &C10, MMUL &C11,
                              const bfloat16 *__restrict a0,
                              const bfloat16 *__restrict a1,
                              const bfloat16 *__restrict b, int colB) {
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

// A pair of side-by-side C tiles: rows 16 wide, split like B.
template <typename T_out>
static inline std::pair<aie::vector<T_out, 32>, aie::vector<T_out, 32>>
load_c(const T_out *__restrict c, int colB) {
  aie::vector<T_out, 64> v =
      aie::concat(aie::load_v<16>(c), aie::load_v<16>(c + colB),
                  aie::load_v<16>(c + 2 * colB), aie::load_v<16>(c + 3 * colB));
  return aie::interleave_unzip(v.template extract<32>(0),
                               v.template extract<32>(1), 8);
}

template <typename T_out>
static inline void store_c(T_out *__restrict c, aie::vector<T_out, 32> v0,
                           aie::vector<T_out, 32> v1, int colB) {
  auto [lo, hi] = aie::interleave_zip(v0, v1, 8);
  aie::store_v(c, lo.template extract<16>(0));
  aie::store_v(c + colB, lo.template extract<16>(1));
  aie::store_v(c + 2 * colB, hi.template extract<16>(0));
  aie::store_v(c + 3 * colB, hi.template extract<16>(1));
}

template <bool get, typename MMUL>
static inline typename MMUL::accum_type own_plus_cascade(const MMUL &C) {
  using Acc = typename MMUL::accum_type;
  Acc acc = C.to_accum();
  if constexpr (get)
    acc = aie::add(acc, cascade_get<Acc>());
  return acc;
}

template <typename T_out, typename Acc>
static inline aie::vector<T_out, 32> add_c(const Acc &acc,
                                           aie::vector<T_out, 32> v) {
  Acc old;
  old.from_vector(v);
  return aie::add(acc, old).template to_vector<T_out>();
}

// Finishes C0 and C1 one at a time: two live accumulators spill.
template <bool get, bool put, typename T_out, typename MMUL>
static inline void cascade_finish(const MMUL &C0, const MMUL &C1,
                                  T_out *__restrict c, int colB) {
  if constexpr (put) {
    cascade_put(own_plus_cascade<get>(C0));
    cascade_put(own_plus_cascade<get>(C1));
  } else {
    auto [v0, v1] = load_c(c, colB);
    v0 = add_c(own_plus_cascade<get>(C0), v0);
    v1 = add_c(own_plus_cascade<get>(C1), v1);
    store_c(c, v0, v1, colB);
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
  constexpr int ks = tile::template ks<colA>;
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
      AIE_LOOP_MIN_ITERATION_COUNT(colA / ks)
      for (int k = 0; k < colA; k += ks) {
        tile::template mac_step<colA>(C00, C01, C10, C11, a0 + k, a1 + k, pb,
                                      colB);
        pb += ks * colB;
      }
      T_out *pc = c + i * colB + j;
      cascade_finish<get, put, T_out>(C00, C01, pc, colB);
      cascade_finish<get, put, T_out>(C10, C11, pc + 4 * colB, colB);
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

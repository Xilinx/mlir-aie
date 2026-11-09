//===- mha.cc ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../activation/softmax_aie2p.h"

// mha.cc is a single compilation unit that includes each arch's own mm
// kernels and the softmax (there is no separate link step).  The col-major B
// variants are compiled by passing -DB_COL_MAJ to the compiler; this flag is
// set in the PeanoCompilationRule configuration for this file.
#include "../aie_kernel_utils.h"
#include "../common/zero.h"
#include "../fused/mm_fused_mmul.h"
#if AIE_ARCH_AIE2
// The products call mm_aie2.h's routines, at block shapes (an 8-row decode
// block) its vectorized entry points do not take; matmul_scalar_bf16_bf16
// stays.
#define SCALAR_ONLY
#include "mm_aie2.h"
#else
// matmul_bf16_bf16 and matmul_scalar_bf16_bf16.
#include "mm_aie2p.h"
#endif

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define VECTOR_LENGTH 64

// The QK^T score tile: float32 under MHA_F32_SCORES, else bf16. A bf16 score
// steps by 2^-8 of its magnitude, which past |A| = 2^15 or so is coarser than
// the softmax can take.
#ifdef MHA_F32_SCORES
using score_t = float;
#else
using score_t = bfloat16;
#endif

#define ROUNDING_MODE aie::rounding_mode::conv_even

// The query at position q keeps the keys k with q - MHA_WINDOW <= k <= q +
// AHEAD: a causal build bounds k by q, a windowed one by the window on either
// side it is not causal on, and one that is neither keeps every key. A window
// of 0 is none. Packed, a block's rows are one query each, of a KV group's
// heads, all at the last key's position, S_kv_eff - 1.
#ifndef MHA_CAUSAL
#define MHA_CAUSAL 1
#endif
#ifndef MHA_WINDOW
#define MHA_WINDOW 0
#endif
#ifndef MHA_PACKED
#define MHA_PACKED 0
#endif
// A block's rows are MHA_GROUP heads of each position, position-major: row
// i is at the block's first position plus i / MHA_GROUP.
#ifndef MHA_GROUP
#define MHA_GROUP 1
#endif
static constexpr bool BAND_AHEAD = MHA_CAUSAL || MHA_WINDOW > 0;
static constexpr int32_t AHEAD = MHA_CAUSAL ? 0 : MHA_WINDOW;
static constexpr bool BAND_BEHIND = MHA_WINDOW > 0;

// idx_buffer is (key block, query block, row offset): the query block's
// first row is at position query block * (B_q / MHA_GROUP) + row offset, so
// the queries may start anywhere. The position of that row less that of the
// key block's.
static inline __attribute__((always_inline)) int32_t block_diagonal(
    const int32_t *idx_buffer, int32_t B_q, int32_t B_kv, int32_t S_kv_eff) {
  return (MHA_PACKED ? S_kv_eff - 1
                     : idx_buffer[1] * (B_q / MHA_GROUP) + idx_buffer[2]) -
         idx_buffer[0] * B_kv;
}

// Whether a query block keeps any key of a key block.
static inline __attribute__((always_inline)) bool
block_live(int32_t diagonal, int32_t B_q, int32_t B_kv) {
  return (!BAND_AHEAD ||
          diagonal + (MHA_PACKED ? 1 : B_q / MHA_GROUP) + AHEAD > 0) &&
         (!BAND_BEHIND || diagonal < MHA_WINDOW + B_kv);
}

// P*V's micro-tile: P (PV_R, PV_S), V (PV_S, PV_T) and O (PV_R, PV_T).
#if AIE_ARCH_AIE2
static constexpr int32_t PV_R = 4, PV_S = 8, PV_T = 4;

// GEMM's AIE2 bf16 product, (M, K) by (K, N) on the native 4x8x4 tile: four
// row blocks by four column blocks a step, or two by two in an 8-row block.
// A column-major B's column blocks are b_pitch tiles apart; c_zero starts
// C at 0 rather than adding to it.
template <typename C, unsigned M, unsigned K, unsigned N, bool b_row_maj,
          unsigned b_pitch = K / 8, bool c_zero = false>
static inline void matmul_aie2(const bfloat16 *a, const bfloat16 *b, C *c) {
  constexpr unsigned r = 4, s = 8, t = 4;
  static_assert(M % (2 * r) == 0 && K % s == 0 && N % (4 * t) == 0);
  if constexpr (M % (4 * r) == 0)
    matmul_vectorized_4x4<bfloat16, C, M / r, K / s, N / t, r, s, t, b_row_maj,
                          true, (K / s) <= 8, b_pitch, c_zero>(a, b, c);
  else
    matmul_vectorized_2x2_mmul<bfloat16, C, M / r, K / s, N / t, r, s, t,
                               b_row_maj, true, false, b_pitch, c_zero>(a, b,
                                                                        c);
}
#else
static constexpr int32_t PV_R = 8, PV_S = 8, PV_T = 8;
#endif

// A vector of O's (PV_R, PV_T) tiles side by side, lane i holding
// s[(i % (PV_R * PV_T)) / PV_T]: one value per row of the tiles.
// aie::transpose stops at 512 bits, so each row's lanes are broadcast.
static inline __attribute__((always_inline)) aie::accum<accfloat, VECTOR_LENGTH>
broadcast_by_row(const float *s) {
  aie::accum<accfloat, VECTOR_LENGTH> v;
#if AIE_ARCH_AIE2
  const auto tile = aie::concat(
      aie::broadcast<float, 4>(s[0]), aie::broadcast<float, 4>(s[1]),
      aie::broadcast<float, 4>(s[2]), aie::broadcast<float, 4>(s[3]));
  v.from_vector(aie::concat(tile, tile, tile, tile));
#else
  v.from_vector(aie::concat(
      aie::broadcast<float, 8>(s[0]), aie::broadcast<float, 8>(s[1]),
      aie::broadcast<float, 8>(s[2]), aie::broadcast<float, 8>(s[3]),
      aie::broadcast<float, 8>(s[4]), aie::broadcast<float, 8>(s[5]),
      aie::broadcast<float, 8>(s[6]), aie::broadcast<float, 8>(s[7])));
#endif
  return v;
}

// Row-major variants needed by matmul_PV.  Because there is no separate link
// step, all kernel symbols must be defined in this single translation unit.

// The product is (M, K) by (K, N) on (PV_R, PV_S, PV_T) tiles. AIE2's is
// GEMM's own routine. AIE2P's native bf16 macs take the shared (half, column
// block) 2x2 tile. Emulated through bfp16 it expands along n only:
// mm_aie2p.h's 2x2 expansion keeps four 64-float accumulators plus two A and
// two B tiles live, which does not fit the registers. The mac order per
// output tile, and so the result, is the same.
template <typename C, unsigned M, unsigned K, unsigned N>
static void matmul_rowmaj(bfloat16 *a_in, bfloat16 *b_in, C *c_out) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
#if AIE_ARCH_AIE2
  matmul_aie2<C, M, K, N, true>(a_in, b_in, c_out);
#else
  constexpr unsigned r = PV_R, s = PV_S, t = PV_T;
  constexpr unsigned rowA = M / r, colA = K / s, colB = N / t;
  static_assert(M % r == 0);
  static_assert(K % s == 0);
  static_assert(N % (2 * t) == 0);
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;

#if !defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16)
  mm_bf16_2x2_aie2p<rowA, colA, colB, r, s, t, false, true>(a_in, b_in, c_out);
#else
  for (unsigned z = 0; z < rowA; z++) {
    C *__restrict pC1 = c_out + (z * colB) * MMUL::size_C;
    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA1 = a_in + (z * colA) * MMUL::size_A;
      const bfloat16 *__restrict pB1 = b_in + j * MMUL::size_B;
      const bfloat16 *__restrict pB2 = b_in + (j + 1) * MMUL::size_B;

      // Accumulates onto C in place; callers zero it when a new accumulation
      // starts.
      MMUL C00(aie::load_v<MMUL::size_C>(pC1));
      MMUL C01(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C));

      for (unsigned i = 0; i < colA; ++i) {
        aie::vector<bfloat16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<bfloat16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B * colB;
        aie::vector<bfloat16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B * colB;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
      }

      aie::store_v(pC1, C00.template to_vector<C>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<C>());
      pC1 += MMUL::size_C;
    }
  }
#endif
#endif
  ::aie::set_rounding(saved_rounding);
}

// QK^T with B^T and at most 8 k steps runs with its reduction unrolled; at 16
// the unrolled body spills, and a row-major B measured slower unrolled.
#if AIE_ARCH_AIE2P && !defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16) &&    \
    !defined(MHA_F32_SCORES) && defined(B_COL_MAJ) && DIM_K <= 64
#define MHA_QK_UNROLL_K 1
#endif

extern "C" {

void matmul_bf16_bf16_rowmaj(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  event0();
  matmul_rowmaj<bfloat16, DIM_M, DIM_K, DIM_N>(a_in, b_in, c_out);
  event1();
}

} // extern "C" (row-major wrappers)

// O is matmul_PV's (PV_R, PV_T) blocked output, B_q rows of the head
// dimension DIM_K: element (row, col) sits at (row / PV_R) * PV_R * DIM_K +
// (col / PV_T) * PV_R * PV_T + (row % PV_R) * PV_T + col % PV_T, so a band
// of PV_R rows is PV_R * DIM_K / 64 vectors. Both O and the scale are three
// limbs; the products under 2^-16 of the leading one are dropped.
static constexpr int32_t BAND_VECTORS = PV_R * DIM_K / VECTOR_LENGTH;
static void scale_blocked_rows(const float *O, bfloat16 *out,
                               const float *scale, int32_t rows) {
  using Acc64 = aie::accum<accfloat, VECTOR_LENGTH>;
  for (int32_t l = 0; l < rows / PV_R; l++) {
    const Limbs<VECTOR_LENGTH> sl =
        bf16_limbs(broadcast_by_row(scale + l * PV_R));
    const float *row = O + l * PV_R * DIM_K;
    bfloat16 *dst = out + l * PV_R * DIM_K;
    for (int32_t j = 0; j < BAND_VECTORS; j++) {
      Acc64 o_acc;
      o_acc.from_vector(aie::load_v<VECTOR_LENGTH>(row + j * VECTOR_LENGTH));
      const Limbs<VECTOR_LENGTH> ol = bf16_limbs(o_acc);
      Acc64 acc = aie::mul(ol.hi, sl.hi);
      acc = aie::mac(acc, ol.hi, sl.mid);
      acc = aie::mac(acc, ol.mid, sl.hi);
      acc = aie::mac(acc, ol.hi, sl.lo);
      acc = aie::mac(acc, ol.mid, sl.mid);
      acc = aie::mac(acc, ol.lo, sl.hi);
      aie::store_v(dst + j * VECTOR_LENGTH, acc.template to_vector<bfloat16>());
    }
  }
}

// Scales O's rows in place by the running max's correction, exact in bf16, so
// only O is split. A row whose max held, a factor of exactly 1, keeps its O:
// through the limbs, an O just below a power of two can lose its last bit.
#if AIE_ARCH_AIE2
static void
#else
// Inlined, as in matmul_PV before attend_block also called it: outlined, it
// changed the composed P*V core's code.
static inline __attribute__((always_inline)) void
#endif
rescale_running_O(float *O, const float *alpha, int32_t rows) {
  using Acc64 = aie::accum<accfloat, VECTOR_LENGTH>;
  // Most key blocks move no row's max: one compare over the block's every
  // factor before the row groups'.
  constexpr int32_t W = DIM_M % 16 ? 8 : 16;
  bool held_all = true;
  AIE_LOOP_UNROLL_FULL
  for (int32_t l = 0; l < DIM_M; l += W)
    held_all &=
        aie::equal(aie::load_v<W>(alpha + l), aie::broadcast<float, W>(1.0f));
  if (held_all)
    return;
  const auto ones = aie::broadcast<float, 8>(1.0f);
  for (int32_t l = 0; l < rows; l += 8) {
    if (aie::equal(aie::load_v<8>(alpha + l), ones))
      continue;
    for (int32_t band = l; band < l + 8; band += PV_R) {
      const Acc64 factors = broadcast_by_row(alpha + band);
      const aie::vector<bfloat16, VECTOR_LENGTH> s =
          factors.template to_vector<bfloat16>();
      const auto held = aie::eq(factors.template to_vector<float>(),
                                aie::broadcast<float, VECTOR_LENGTH>(1.0f));
      float *__restrict row = O + band * DIM_K;
      // Rolled: the fused cores' 16 KB of program memory, f32's in particular.
      AIE_LOOP_NO_UNROLL
      for (int32_t j = 0; j < BAND_VECTORS; j++) {
        const aie::vector<float, VECTOR_LENGTH> o =
            aie::load_v<VECTOR_LENGTH>(row);
        Acc64 o_acc;
        o_acc.from_vector(o);
        const Limbs<VECTOR_LENGTH> ol = bf16_limbs(o_acc);
        Acc64 acc = aie::mul(ol.hi, s);
        acc = aie::mac(acc, ol.mid, s);
        acc = aie::mac(acc, ol.lo, s);
        aie::store_v(row,
                     aie::select(acc.template to_vector<float>(), o, held));
        row += VECTOR_LENGTH;
      }
    }
  }
}

// partial_softmax_alias_bf16 over every valid row of a block, a row a key
// block of ROW keys, reduced eight rows at a time: unzipping two partially
// reduced vectors pairs each row's low half with its high half, reduce_max's
// and reduce_add's pairing, so P and scale_buffer match the per-row call bit
// for bit. Masked lanes are set to lowest in place first. The max is taken
// before scaling, which a positive scale leaves unchanged.
static constexpr int32_t ROW = DIM_N;
static_assert(ROW == 16 || ROW == 32 || ROW == 64,
              "a key block is 16, 32 or 64 keys: one vector a row");
static constexpr int32_t SM_ROWS = 8;
static constexpr int32_t SM_LANES = 16;
#if DIM_M == 8
#define MHA_GROUP_LOOP
#else
#define MHA_GROUP_LOOP AIE_LOOP_MIN_ITERATION_COUNT(2)
#endif

// Lane numbers, compared against the ends of a row's band, build its mask in a
// vector compare or two.
alignas(64) static const int16_t sm_lane_idx[VECTOR_LENGTH] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

// The lanes row i discards: at position p = i / MHA_GROUP it keeps the
// columns from p + diagonal - MHA_WINDOW to p + diagonal + AHEAD that are
// before valid_cols, packed rows all those of row 0.
template <unsigned W>
static inline __attribute__((always_inline)) aie::mask<W>
band_mask(aie::vector<int16_t, W> lane, int32_t i, int32_t valid_cols,
          int32_t diagonal) {
  i = MHA_PACKED ? 0 : i / MHA_GROUP;
  int32_t end = valid_cols;
  if (BAND_AHEAD && i + diagonal + AHEAD + 1 < end)
    end = i + diagonal + AHEAD + 1;
  aie::mask<W> drop = aie::ge(lane, aie::broadcast<int16_t, W>(end));
  if constexpr (BAND_BEHIND) {
    const int32_t start = i + diagonal - MHA_WINDOW;
    drop =
        drop | aie::lt(lane, aie::broadcast<int16_t, W>(start < 0 ? 0 : start));
  }
  return drop;
}

// at(r) is row r narrowed to W lanes, and lane r of the result is row r.
// Every lane of the result depends only on its own row, so rows past the last
// valid one may hold anything.
template <typename T, unsigned W, typename Load, typename Fold>
static inline __attribute__((always_inline)) aie::vector<T, W>
fold_rows(Load at, Fold fold) {
  aie::vector<T, W> q0 =
      fold(fold(at(0), at(1), W / 2), fold(at(2), at(3), W / 2), W / 4);
  aie::vector<T, W> q1 =
      fold(fold(at(4), at(5), W / 2), fold(at(6), at(7), W / 2), W / 4);
  aie::vector<T, W> o = fold(q0, q1, W / 8);
  AIE_LOOP_UNROLL_FULL
  for (unsigned step = W / 16; step > 0; step /= 2)
    o = fold(o, o, step);
  return o;
}

#if AIE_TUNED_AIE2 && !defined(EXP2_BF16_ACCURATE)
static inline aie::vector<bfloat16, SM_LANES>
exp2_neg_reduced(aie::accum<accfloat, SM_LANES> f,
                 aie::vector<int32_t, SM_LANES> k);

// 2^-y for y = m * s - a * s, 16 lanes, from x = [a | m] and neg_scale =
// [-s | s]: AIE2's bf16 product sums a lane of each half.  exp2_bf16.h's
// reduction and Horner steps in -f, each constant term an f32, so the two
// agree bit for bit, but k = round(y) is not clamped.  Shifting it into the
// exponent field saturates instead (the caller sets saturation), and maxdiff
// clamps the difference at +0, so every y above 128, masked lanes included,
// gives +0 whatever 2^-f came to.  Below -127.5, where 2^-y overflows, the
// result is undefined; m is the row's largest a and both products are exact, so
// a live row's y is at least 0.
static inline __attribute__((always_inline)) aie::vector<bfloat16, SM_LANES>
exp2_neg_bf16(aie::vector<bfloat16, 2 * SM_LANES> x,
              aie::vector<bfloat16, 2 * SM_LANES> neg_scale,
              aie::accum<accfloat, SM_LANES> magic) {
  using Acc = aie::accum<accfloat, SM_LANES>;
  // The mac rounds after each product, so y + magic is not one mac: that
  // rounds m - k' + magic with k' = round(a * s).
  Acc y(mul_elem_16_2(x, neg_scale));
  const aie::vector<float, SM_LANES> ym =
      aie::add(magic, y.to_vector<float>()).to_vector<float>();
  const aie::vector<int32_t, SM_LANES> k = aie::sub(
      ym.cast_to<int32_t>(), aie::broadcast<int32_t, SM_LANES>(0x4b400000));
  Acc f(mac_elem_16_2(x, neg_scale, aie::sub(magic, ym)));
  return exp2_neg_reduced(f, k);
}

// 2^-y from y = k + f, k = round(y) and f in [-1/2, 1/2]: exp2_neg_bf16's
// Horner steps and its exponent shift.
static inline __attribute__((always_inline)) aie::vector<bfloat16, SM_LANES>
exp2_neg_reduced(aie::accum<accfloat, SM_LANES> f,
                 aie::vector<int32_t, SM_LANES> k) {
  using Acc = aie::accum<accfloat, SM_LANES>;
  const auto fn = f.to_vector<bfloat16>();
  auto horner = [&](aie::vector<bfloat16, SM_LANES> c1, float c0) {
    Acc acc;
    acc.from_vector(aie::broadcast<float, SM_LANES>(c0));
    return aie::mac(acc, fn, c1);
  };
  Acc t =
      horner(aie::broadcast<bfloat16, SM_LANES>(-0.0555041087f), 0.2402265069f);
  t = horner(t.to_vector<bfloat16>(), -0.6931471805f);
  t = horner(t.to_vector<bfloat16>(), 1.0f);
  Acc r;
  r.from_vector(
      aie::maxdiff(t.to_vector<float>().cast_to<int32_t>(), aie::upshift(k, 23))
          .cast_to<float>());
  return r.to_vector<bfloat16>();
}

// 2^-y for an accumulated y: exp2_neg_bf16's reduction with f = y - k taken
// from y itself, so bit for bit exp2_bf16(-y) over a live row's y >= 0.
static inline __attribute__((always_inline)) aie::vector<bfloat16, SM_LANES>
exp2_neg_f32(aie::accum<accfloat, SM_LANES> y,
             aie::accum<accfloat, SM_LANES> magic) {
  const aie::vector<float, SM_LANES> ym =
      aie::add(magic, y.to_vector<float>()).to_vector<float>();
  const aie::vector<int32_t, SM_LANES> k = aie::sub(
      ym.cast_to<int32_t>(), aie::broadcast<int32_t, SM_LANES>(0x4b400000));
  return exp2_neg_reduced(aie::add(aie::sub(magic, ym), y.to_vector<float>()),
                          k);
}
#endif

// T is the score type: bfloat16, or float under MHA_F32_SCORES.
template <typename T>
static void
partial_softmax_rows(T *__restrict A, bfloat16 *__restrict P,
                     const float *__restrict m_prev, float *__restrict m_new,
                     float *__restrict l_new, int32_t rows, int32_t valid_cols,
                     int32_t diagonal, bfloat16 scale) {
  // A row's max folds at most SM_VEC_LEN lanes, a 64-key row's two halves
  // first; its sum, halved at least once, parks in the row's own A bytes.
  constexpr int32_t FOLD = ROW < SM_VEC_LEN ? ROW : SM_VEC_LEN;
  constexpr int32_t PARK = std::min<int32_t>(
      {ROW / 2, SM_LANES, ROW * (int32_t)sizeof(T) / (int32_t)sizeof(float)});
  const auto scale_vec = aie::broadcast<bfloat16, ROW>(scale);
  const auto lowest_vec =
      aie::broadcast<T, ROW>((T)std::numeric_limits<bfloat16>::lowest());
  const auto lane = aie::load_v<ROW>(sm_lane_idx);
  auto fold_max = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::max(lo, hi);
  };
  auto fold_add = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::add(aie::accum<accfloat, decltype(a)::size()>(lo), hi)
        .template to_vector<float>();
  };

  // The row passes run on to a whole group: the extra rows are padding that
  // was filled with lowest, and the caller zeroes their P rows after.
  const int32_t group_rows = (rows + SM_ROWS - 1) & ~(SM_ROWS - 1);

  // Lanes a row discards hold lowest from here on.
  if ((BAND_AHEAD && diagonal + AHEAD < ROW - 1) || valid_cols < ROW ||
      (BAND_BEHIND &&
       (MHA_PACKED ? 0 : (group_rows - 1) / MHA_GROUP) + diagonal >
           MHA_WINDOW)) {
    T *__restrict row = A;
    AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
    AIE_LOOP_UNROLL(4)
    for (int32_t i = 0; i < group_rows; i++) {
      aie::store_v(row, aie::select(aie::load_v<ROW>(row), lowest_vec,
                                    band_mask(lane, i, valid_cols, diagonal)));
      row += ROW;
    }
  }

  // Rows past the last valid one keep their scale_buffer entries; their P rows
  // are zeroed after. The group loops run over at least two groups, which the
  // caller's block always has, so that the pipeliner may overlap them; an
  // 8-row build's block is one group.
  const int32_t group_end = DIM_M == SM_ROWS       ? SM_ROWS
                            : group_rows > SM_ROWS ? group_rows
                                                   : 2 * SM_ROWS;
  auto live = [rows](int32_t g) {
    int32_t n = rows - g >= SM_ROWS ? SM_ROWS : rows - g > 0 ? rows - g : 0;
    return aie::mask<SM_ROWS>::from_uint32((1u << n) - 1);
  };
  // The maxima are of the unscaled scores, exact in bf16, and are scaled in
  // the exponentials' own products, so a row's largest argument is exactly 0.
  // A max scaled and rounded to bf16 can sit half an ulp (128 from 2^15 on)
  // below its element's product, and 2^128 overflows.
  alignas(32) T m_rows[DIM_M];
  const T *__restrict a = A;
  // AIE2 has no float max. A float row's largest is its bits' largest as
  // int32 where that is a non-negative float, else (every float negative)
  // their smallest; the few lanes after are ordered by keys, a negative's
  // magnitude bits flipped. Both exact.
  constexpr bool bits_max = std::is_same_v<T, float> && AIE_ARCH_AIE2;
  using K = std::conditional_t<bits_max, int32_t, T>;
  auto key = [](auto v) {
    if constexpr (bits_max) {
      const auto b = v.template cast_to<int32_t>();
      constexpr unsigned N = decltype(b)::size();
      return aie::select(
          b, aie::bit_xor(b, aie::broadcast<int32_t, N>(0x7fffffff)),
          aie::lt(b, aie::zeros<int32_t, N>()));
    } else {
      return v;
    }
  };
  auto fold_min = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::min(lo, hi);
  };
  // A row's bits as int32, its 16-lane pieces combined by pick first, so
  // the folds across rows stay in one register each.
  auto bits_fold = [&](auto fold, auto pick) {
    return fold_rows<int32_t, SM_LANES>(
               [&](int32_t r) {
                 const T *row = a + r * ROW;
                 auto v =
                     aie::load_v<SM_LANES>(row).template cast_to<int32_t>();
                 AIE_LOOP_UNROLL_FULL
                 for (int32_t c = SM_LANES; c < ROW; c += SM_LANES)
                   v = pick(v, aie::load_v<SM_LANES>(row + c)
                                   .template cast_to<int32_t>());
                 return v;
               },
               fold)
        .template extract<SM_ROWS>(0);
  };
  MHA_GROUP_LOOP
  for (int32_t g = 0; g < group_end; g += SM_ROWS) {
    aie::vector<K, SM_ROWS> o_key;
    if constexpr (bits_max) {
      auto bits =
          bits_fold(fold_max, [](auto x, auto y) { return aie::max(x, y); });
      // A row whose every score is negative (padding, a row masked whole)
      // takes its smallest bits.
      const auto zero = aie::zeros<int32_t, SM_ROWS>();
      if (aie::lt(bits, zero).to_uint32())
        bits = aie::select(
            bits_fold(fold_min, [](auto x, auto y) { return aie::min(x, y); }),
            bits, aie::ge(bits, zero));
      o_key = key(bits.template cast_to<float>());
    } else {
      o_key = fold_rows<T, FOLD>(
                  [&](int32_t r) {
                    if constexpr (ROW == FOLD) {
                      return aie::load_v<ROW>(a + r * ROW);
                    } else {
                      auto row = aie::load_v<ROW>(a + r * ROW);
                      return aie::max(row.template extract<FOLD>(0),
                                      row.template extract<FOLD>(1));
                    }
                  },
                  fold_max)
                  .template extract<SM_ROWS>(0);
    }
    a += SM_ROWS * ROW;
    aie::accum<accfloat, SM_ROWS> prev, kept;
    prev.from_vector(aie::load_v<SM_ROWS>(m_prev + g));
    kept.from_vector(aie::load_v<SM_ROWS>(m_new + g));
    const auto top_key =
        aie::max(aie::max(key(lowest_vec.template extract<SM_ROWS>(0)), o_key),
                 key(prev.to_vector<T>()));
    aie::vector<T, SM_ROWS> top;
    if constexpr (bits_max)
      top = key(top_key).template cast_to<float>();
    else
      top = top_key;
    auto m = aie::select(kept.to_vector<T>(), top, live(g));
    aie::store_v(m_rows + g, m);
    aie::store_v(m_new + g,
                 aie::accum<accfloat, SM_ROWS>(m).to_vector<float>());
  }

  // Each row's exponentials go to P.  A discarded lane's is +0.
  bfloat16 *__restrict p = P;
  if constexpr (std::is_same_v<T, float>) {
#if AIE_TUNED_AIE2 && !defined(EXP2_BF16_ACCURATE)
    // (m - A) * s from m - A's three bf16 limbs, the first two in one
    // product sum (rounded once, as a mul then a mac round it), then
    // exp2_neg_f32: bit for bit exp2_bf16 of the generic path's (A - m) * s.
    const auto s2 = aie::broadcast<bfloat16, 2 * SM_LANES>(scale);
    const auto zero = aie::zeros<bfloat16, SM_LANES>();
    aie::accum<accfloat, SM_LANES> magic;
    magic.from_vector(aie::broadcast<float, SM_LANES>(12582912.0f));
    aie::saturation_mode saved_saturation =
        aie::swap_saturation(aie::saturation_mode::saturate);
    // As the bf16 path's loop: 32 lanes a step, each row's own maximum.
    constexpr int32_t STEP_ROWS = ROW < 2 * SM_LANES ? 2 * SM_LANES / ROW : 1;
    auto half = [&](const float *src, float m) {
      aie::accum<accfloat, SM_LANES> m_acc;
      m_acc.from_vector(aie::broadcast<float, SM_LANES>(m));
      const Limbs<SM_LANES> dl =
          bf16_limbs(aie::sub(m_acc, aie::load_v<SM_LANES>(src)));
      aie::accum<accfloat, SM_LANES> y(
          mul_elem_16_2(aie::concat(dl.hi, dl.mid), s2));
      y = mac_elem_16_2(aie::concat(dl.lo, zero), s2, y);
      return exp2_neg_f32(y, magic);
    };
    for (int32_t h = 0; h < ROW; h += 2 * SM_LANES) {
      a = A + h;
      p = P + h;
      const T *__restrict m = m_rows;
      AIE_LOOP_MIN_ITERATION_COUNT(4)
      for (int32_t i = 0; i < group_rows; i += STEP_ROWS) {
        const float m_lo = m[0], m_hi = m[STEP_ROWS - 1];
        m += STEP_ROWS;
        aie::store_v(p, half(a, m_lo));
        aie::store_v(p + SM_LANES, half(a + SM_LANES, m_hi));
        a += ROW * STEP_ROWS;
        p += ROW * STEP_ROWS;
      }
    }
    aie::set_saturation(saved_saturation);
#elif AIE_ARCH_AIE2P
    // As below, 64 lanes a step, a row's or several rows' each against its
    // own row's maximum, so that two exponential chains share a schedule.
    constexpr int32_t LANES = 64;
    constexpr int32_t STEP_ROWS = LANES / ROW;
    const auto s = aie::broadcast<bfloat16, LANES>(scale);
    a = A;
    AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS / STEP_ROWS)
    for (int32_t i = 0; i < group_rows; i += STEP_ROWS) {
      aie::vector<float, LANES> m_vec;
      AIE_LOOP_UNROLL_FULL
      for (int32_t r = 0; r < STEP_ROWS; r++)
        m_vec.insert(r, aie::broadcast<float, ROW>(m_rows[i + r]));
      aie::accum<accfloat, LANES> d;
      d.from_vector(aie::load_v<LANES>(a));
      const Limbs<LANES> dl = bf16_limbs(aie::sub(d, m_vec));
      aie::accum<accfloat, LANES> x = aie::mul(dl.hi, s);
      x = aie::mac(x, dl.mid, s);
      x = aie::mac(x, dl.lo, s);
      aie::store_v(p, exp2_bf16(x.to_vector<float>()));
      a += LANES;
      p += LANES;
    }
#else
    // A float32 max leaves A - m exact in the accumulator, so (A - m) * s is
    // taken as three bf16 limbs, each times s.
    const auto s = aie::broadcast<bfloat16, FOLD>(scale);
    a = A;
    AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
    for (int32_t i = 0; i < group_rows; i++) {
      const auto m_vec = aie::broadcast<float, FOLD>(m_rows[i]);
      for (int32_t h = 0; h < ROW; h += FOLD) {
        aie::accum<accfloat, FOLD> d;
        d.from_vector(aie::load_v<FOLD>(a + h));
        const Limbs<FOLD> dl = bf16_limbs(aie::sub(d, m_vec));
        aie::accum<accfloat, FOLD> x = aie::mul(dl.hi, s);
        x = aie::mac(x, dl.mid, s);
        x = aie::mac(x, dl.lo, s);
        aie::store_v(p + h, exp2_bf16(x.to_vector<float>()));
      }
      a += ROW;
      p += ROW;
    }
#endif
  } else {
    const T *__restrict m = m_rows;
#if AIE_TUNED_AIE2 && !defined(EXP2_BF16_ACCURATE)
    // Half a 64-key row per pass keeps the loop within the registers; a
    // whole row spills.
    const auto neg_scale =
        aie::concat(aie::broadcast<bfloat16, SM_LANES>(-scale),
                    aie::broadcast<bfloat16, SM_LANES>(scale));
    aie::accum<accfloat, SM_LANES> magic;
    magic.from_vector(aie::broadcast<float, SM_LANES>(12582912.0f));
    aie::saturation_mode saved_saturation =
        aie::swap_saturation(aie::saturation_mode::saturate);
    // A step takes 32 lanes: half a 64-key row, a 32-key row, or two 16-key
    // rows, each with its own row's maximum.
    constexpr int32_t STEP_ROWS = ROW < 2 * SM_LANES ? 2 * SM_LANES / ROW : 1;
    for (int32_t h = 0; h < ROW; h += 2 * SM_LANES) {
      a = A + h;
      p = P + h;
      m = m_rows;
      AIE_LOOP_MIN_ITERATION_COUNT(4)
      for (int32_t i = 0; i < group_rows; i += STEP_ROWS) {
        const auto m_lo = aie::broadcast<bfloat16, SM_LANES>(m[0]);
        const auto m_hi = aie::broadcast<bfloat16, SM_LANES>(m[STEP_ROWS - 1]);
        m += STEP_ROWS;
        aie::store_v(p,
                     exp2_neg_bf16(aie::concat(aie::load_v<SM_LANES>(a), m_lo),
                                   neg_scale, magic));
        aie::store_v(p + SM_LANES,
                     exp2_neg_bf16(
                         aie::concat(aie::load_v<SM_LANES>(a + SM_LANES), m_hi),
                         neg_scale, magic));
        a += ROW * STEP_ROWS;
        p += ROW * STEP_ROWS;
      }
    }
    aie::set_saturation(saved_saturation);
#else
    // Taking m off as m * s in the multiplier subtracts the product the row's
    // largest element takes, without widening m to ROW accumulator lanes
    // first.
    a = A;
    AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
#ifndef EXP2_BF16_ACCURATE
    AIE_LOOP_UNROLL(2)
#endif
    for (int32_t i = 0; i < group_rows; i++) {
      aie::accum<accfloat, ROW> exp_in =
          aie::msc(aie::mul(aie::load_v<ROW>(a), scale_vec),
                   aie::broadcast<bfloat16, ROW>(*m++), scale_vec);
      aie::store_v(p, exp2_bf16(exp_in.to_vector<float>()));
      a += ROW;
      p += ROW;
    }
#endif
  }

  // Their sums, halved to PARK lanes as reduce_add pairs them, are parked in
  // the A rows, which nothing reads after this call.  Doing this in the loop
  // above would make each row's load wait on the previous row's store.
  p = P;
  T *__restrict sums = A;
  AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
  AIE_LOOP_UNROLL(2)
  for (int32_t i = 0; i < group_rows; i++) {
    auto exp_val = aie::load_v<ROW>(p);
    p += ROW;
    // No lane is -0, so starting from the first half is starting from zero.
    aie::accum<accfloat, ROW / 2> sum(exp_val.template extract<ROW / 2>(0));
    auto s = aie::add(sum, exp_val.template extract<ROW / 2>(1))
                 .template to_vector<float>();
    aie::vector<float, PARK> half;
    if constexpr (ROW / 2 == PARK)
      half = s;
    else
      half = aie::add(aie::accum<accfloat, PARK>(s.template extract<PARK>(0)),
                      s.template extract<PARK>(1))
                 .template to_vector<float>();
    aie::store_v(sums, aie::vector_cast<T>(half));
    sums += ROW;
  }

  sums = A;
  MHA_GROUP_LOOP
  for (int32_t g = 0; g < group_end; g += SM_ROWS) {
    aie::vector<float, PARK> t = fold_rows<float, PARK>(
        [&](int32_t r) {
          return aie::vector_cast<float>(
              aie::load_v<PARK * sizeof(float) / sizeof(T)>(sums + r * ROW));
        },
        fold_add);
    sums += SM_ROWS * ROW;
    aie::store_v(l_new + g,
                 aie::select(aie::load_v<SM_ROWS>(l_new + g),
                             t.template extract<SM_ROWS>(0), live(g)));
  }
}

#if (AIE_TUNED_AIE2 &&                                                         \
     (!defined(EXP2_BF16_ACCURATE) || defined(MHA_F32_SCORES))) ||             \
    (AIE_TUNED_AIE2P && defined(EXP2_BF16_ACCURATE) &&                         \
     !defined(AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16))
#define MHA_FUSED 1
#if AIE_ARCH_AIE2
// The fused block's softmax: partial_softmax_rows over S as QK^T leaves it,
// (4, 4) tiles in row-major tile order, writing P as P*V reads it, (4, 8)
// tiles, each the two S tiles it covers; P apart from S, so the exponentials
// pipeline. P and the state are
// partial_softmax_rows' bit for bit: a max is exact in any order, a weight
// is its own lane's, and a row's sum adds the same pairs, keys half the row
// apart, then half that, down to neighbours.
static constexpr int32_t S_TILE = 16;
static constexpr int32_t BAND = 4 * ROW;
static constexpr int32_t S_TILES = ROW / 4;
static constexpr int32_t P_TILES = ROW / 8;
static_assert(PV_R == 4 && PV_S == 8, "a P tile is two S tiles side by side");

alignas(64) static const int32_t s_tile_key[S_TILE] = {0, 1, 2, 3, 0, 1, 2, 3,
                                                       0, 1, 2, 3, 0, 1, 2, 3};

// Four values, each over its row's four lanes of an S tile; 32-bit lanes,
// AIE2's narrowest four.
template <typename T>
static inline __attribute__((always_inline)) aie::vector<T, S_TILE>
by_tile_row(T v0, T v1, T v2, T v3) {
  return aie::concat(aie::broadcast<T, 4>(v0), aie::broadcast<T, 4>(v1),
                     aie::broadcast<T, 4>(v2), aie::broadcast<T, 4>(v3));
}

// S is the score type's, as QK^T writes it: bf16, or float under
// MHA_F32_SCORES, whose maxima and exponentials are partial_softmax_rows'.
template <typename T>
static void
partial_softmax_tiles(T *__restrict S, bfloat16 *__restrict P,
                      const float *__restrict m_prev, float *__restrict m_new,
                      float *__restrict l_new, int32_t rows, int32_t valid_cols,
                      int32_t diagonal, bfloat16 scale) {
  const auto lowest16 =
      aie::broadcast<T, S_TILE>((T)std::numeric_limits<bfloat16>::lowest());
  auto fold_max = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::max(lo, hi);
  };
  auto fold_min = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::min(lo, hi);
  };
  constexpr bool f32 = std::is_same_v<T, float>;
  // As partial_softmax_rows: a float maximum through int32 bits and keys.
  auto key = [](auto v) {
    const auto b = v.template cast_to<int32_t>();
    constexpr unsigned N = decltype(b)::size();
    return aie::select(b,
                       aie::bit_xor(b, aie::broadcast<int32_t, N>(0x7fffffff)),
                       aie::lt(b, aie::zeros<int32_t, N>()));
  };
  auto fold_add = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::add(aie::accum<accfloat, decltype(a)::size()>(lo), hi)
        .template to_vector<float>();
  };
  auto live = [rows](int32_t g) {
    int32_t n = rows - g >= SM_ROWS ? SM_ROWS : rows - g > 0 ? rows - g : 0;
    return aie::mask<SM_ROWS>::from_uint32((1u << n) - 1);
  };

  // Lanes a row discards, and a padded row's every lane, hold lowest from
  // here on, as partial_softmax and band_mask leave them.
  if ((BAND_AHEAD && diagonal + AHEAD < ROW - 1) || valid_cols < ROW ||
      (BAND_BEHIND &&
       (MHA_PACKED ? 0 : (DIM_M - 1) / MHA_GROUP) + diagonal > MHA_WINDOW) ||
      rows < DIM_M) {
    const auto key0 = aie::load_v<S_TILE>(s_tile_key);
    for (int32_t b = 0; b < DIM_M / 4; b++) {
      int32_t end[4], start[4];
      for (int32_t r = 0; r < 4; r++) {
        const int32_t i = b * 4 + r;
        const int32_t pos = MHA_PACKED ? 0 : i / MHA_GROUP;
        end[r] = valid_cols;
        if (BAND_AHEAD && pos + diagonal + AHEAD + 1 < end[r])
          end[r] = pos + diagonal + AHEAD + 1;
        start[r] = BAND_BEHIND ? pos + diagonal - MHA_WINDOW : 0;
        if (i >= rows)
          end[r] = 0;
      }
      // Each against the tile's first key, which steps by 4.
      auto end_v = by_tile_row<int32_t>(end[0], end[1], end[2], end[3]);
      auto start_v =
          by_tile_row<int32_t>(start[0], start[1], start[2], start[3]);
      const auto four = aie::broadcast<int32_t, S_TILE>(4);
      T *__restrict t = S + b * BAND;
      for (int32_t c = 0; c < S_TILES; c++, t += S_TILE) {
        aie::mask<S_TILE> drop = aie::ge(key0, end_v);
        if constexpr (BAND_BEHIND)
          drop = drop | aie::lt(key0, start_v);
        aie::store_v(t, aie::select(aie::load_v<S_TILE>(t), lowest16, drop));
        end_v = aie::sub(end_v, four);
        start_v = aie::sub(start_v, four);
      }
    }
  }

  // Each row's maximum, of the unscaled scores, eight rows (two bands) at a
  // time, and each band's over its rows' four lanes of an S tile.
  alignas(32) T m_tiles[DIM_M * 4];
  MHA_GROUP_LOOP
  for (int32_t g = 0; g < DIM_M; g += SM_ROWS) {
    aie::vector<T, SM_ROWS> m;
    aie::accum<accfloat, SM_ROWS> prev, kept;
    prev.from_vector(aie::load_v<SM_ROWS>(m_prev + g));
    kept.from_vector(aie::load_v<SM_ROWS>(m_new + g));
    if constexpr (f32) {
      // A float band's bits, their largest and smallest over its tiles in one
      // pass, then over its rows' lanes; the smallest is a row's where every
      // score is negative.
      aie::vector<int32_t, S_TILE> hi[2], lo[2];
      AIE_LOOP_UNROLL_FULL
      for (int32_t h = 0; h < 2; h++) {
        const T *__restrict t = S + (g / 4 + h) * BAND;
        hi[h] = lo[h] = aie::load_v<S_TILE>(t).template cast_to<int32_t>();
        // Unrolled whole, a 64-key block's 15 tiles spill ~1.6 KB of stack.
        AIE_LOOP_UNROLL(5)
        for (int32_t c = 1; c < S_TILES; c++) {
          const auto v =
              aie::load_v<S_TILE>(t + c * S_TILE).template cast_to<int32_t>();
          hi[h] = aie::max(hi[h], v);
          lo[h] = aie::min(lo[h], v);
        }
      }
      const auto hi_pairs = fold_max(hi[0], hi[1], 1);
      const auto lo_pairs = fold_min(lo[0], lo[1], 1);
      const auto zero = aie::zeros<int32_t, SM_ROWS>();
      const auto top_bits =
          fold_max(hi_pairs, hi_pairs, 1).template extract<SM_ROWS>(0);
      const auto bits = aie::select(
          fold_min(lo_pairs, lo_pairs, 1).template extract<SM_ROWS>(0),
          top_bits, aie::ge(top_bits, zero));
      const auto top =
          aie::max(aie::max(key(lowest16.template extract<SM_ROWS>(0)),
                            key(bits.template cast_to<float>())),
                   key(prev.to_vector<float>()));
      m = aie::select(kept.to_vector<float>(),
                      key(top).template cast_to<float>(), live(g));
    } else {
      aie::vector<bfloat16, S_TILE> band_max[2];
      AIE_LOOP_UNROLL_FULL
      for (int32_t h = 0; h < 2; h++) {
        const bfloat16 *__restrict t = S + (g / 4 + h) * BAND;
        auto mx = aie::load_v<2 * S_TILE>(t);
        AIE_LOOP_UNROLL_FULL
        for (int32_t c = 2; c < S_TILES; c += 2)
          mx = aie::max(mx, aie::load_v<2 * S_TILE>(t + c * S_TILE));
        band_max[h] = aie::max(mx.template extract<S_TILE>(0),
                               mx.template extract<S_TILE>(1));
      }
      const auto pairs = fold_max(band_max[0], band_max[1], 1);
      const aie::vector<bfloat16, SM_ROWS> o =
          fold_max(pairs, pairs, 1).template extract<SM_ROWS>(0);
      m = aie::select(
          kept.to_vector<bfloat16>(),
          aie::max(aie::max(lowest16.template extract<SM_ROWS>(0), o),
                   prev.to_vector<bfloat16>()),
          live(g));
    }
    aie::store_v(m_new + g,
                 aie::accum<accfloat, SM_ROWS>(m).to_vector<float>());
    // Each row's maximum four times over: two zips of m with itself.
    const auto [twice_lo, twice_hi] = aie::interleave_zip(m, m, 1);
    AIE_LOOP_UNROLL_FULL
    for (int32_t h = 0; h < 2; h++) {
      const auto twice = h ? twice_hi : twice_lo;
      const auto [lo, hi] = aie::interleave_zip(twice, twice, 2);
      aie::store_v(m_tiles + (g / 4 + h) * S_TILE, aie::concat(lo, hi));
    }
  }

  // Each tile's exponentials, two S tiles zipped into the P tile over them.
  const auto neg_scale = aie::concat(aie::broadcast<bfloat16, SM_LANES>(-scale),
                                     aie::broadcast<bfloat16, SM_LANES>(scale));
  aie::accum<accfloat, SM_LANES> magic;
  magic.from_vector(aie::broadcast<float, SM_LANES>(12582912.0f));
  aie::saturation_mode saved_saturation =
      aie::swap_saturation(aie::saturation_mode::saturate);
  const T *__restrict t = S;
  bfloat16 *__restrict w = P;
  if constexpr (f32) {
#ifdef EXP2_BF16_ACCURATE
    // (S - m) * s from three bf16 limbs, then the accurate exp2_bf16.
    aie::set_saturation(saved_saturation);
    const auto s32 = aie::broadcast<bfloat16, 2 * S_TILE>(scale);
    AIE_LOOP_MIN_ITERATION_COUNT(DIM_M / 4 * P_TILES)
    for (int32_t i = 0; i < DIM_M / 4 * P_TILES;
         i++, t += 2 * S_TILE, w += 2 * S_TILE) {
      const auto m_v = aie::load_v<S_TILE>(m_tiles + i / P_TILES * S_TILE);
      aie::accum<accfloat, 2 * S_TILE> d;
      d.from_vector(aie::load_v<2 * S_TILE>(t));
      const Limbs<2 * S_TILE> dl =
          bf16_limbs(aie::sub(d, aie::concat(m_v, m_v)));
      aie::accum<accfloat, 2 * S_TILE> x = aie::mul(dl.hi, s32);
      x = aie::mac(x, dl.mid, s32);
      x = aie::mac(x, dl.lo, s32);
      const auto e = exp2_bf16(x.template to_vector<float>());
      const auto [z0, z1] = aie::interleave_zip(
          e.template extract<S_TILE>(0), e.template extract<S_TILE>(1), 4);
      aie::store_v(w, aie::concat(z0, z1));
    }
#else
    // Each lane against its own row's maximum.
    const auto s2 = aie::broadcast<bfloat16, 2 * SM_LANES>(scale);
    const auto zero = aie::zeros<bfloat16, SM_LANES>();
    auto exps = [&](const float *src, aie::vector<float, S_TILE> m) {
      aie::accum<accfloat, SM_LANES> m_acc;
      m_acc.from_vector(m);
      const Limbs<SM_LANES> dl =
          bf16_limbs(aie::sub(m_acc, aie::load_v<SM_LANES>(src)));
      aie::accum<accfloat, SM_LANES> y(
          mul_elem_16_2(aie::concat(dl.hi, dl.mid), s2));
      y = mac_elem_16_2(aie::concat(dl.lo, zero), s2, y);
      return exp2_neg_f32(y, magic);
    };
    AIE_LOOP_MIN_ITERATION_COUNT(DIM_M / 4 * P_TILES)
    for (int32_t i = 0; i < DIM_M / 4 * P_TILES;
         i++, t += 2 * S_TILE, w += 2 * S_TILE) {
      const auto m_v = aie::load_v<S_TILE>(m_tiles + i / P_TILES * S_TILE);
      const auto [z0, z1] =
          aie::interleave_zip(exps(t, m_v), exps(t + S_TILE, m_v), 4);
      aie::store_v(w, aie::concat(z0, z1));
    }
#endif
  } else {
#ifndef EXP2_BF16_ACCURATE
    AIE_LOOP_MIN_ITERATION_COUNT(DIM_M / 4 * P_TILES)
    for (int32_t i = 0; i < DIM_M / 4 * P_TILES;
         i++, t += 2 * S_TILE, w += 2 * S_TILE) {
      const auto m_v = aie::load_v<S_TILE>(m_tiles + i / P_TILES * S_TILE);
      const auto s = aie::load_v<2 * S_TILE>(t);
      const auto e0 = exp2_neg_bf16(
          aie::concat(s.template extract<S_TILE>(0), m_v), neg_scale, magic);
      const auto e1 = exp2_neg_bf16(
          aie::concat(s.template extract<S_TILE>(1), m_v), neg_scale, magic);
      const auto [z0, z1] = aie::interleave_zip(e0, e1, 4);
      aie::store_v(w, aie::concat(z0, z1));
    }
#endif
  }
  aie::set_saturation(saved_saturation);

  // Each row's sum over its P tiles, eight rows at a time: tiles half the row
  // apart, then half that, to one tile, then lanes four, two and one apart.
  MHA_GROUP_LOOP
  for (int32_t g = 0; g < DIM_M; g += SM_ROWS) {
    aie::vector<float, S_TILE> quarter[2];
    for (int32_t h = 0; h < 2; h++) {
      const bfloat16 *__restrict t = P + (g / 4 + h) * BAND;
      aie::vector<float, 2 * S_TILE> part[P_TILES / 2 > 0 ? P_TILES / 2 : 1];
      AIE_LOOP_UNROLL_FULL
      for (int32_t k = 0; k < P_TILES / 2; k++)
        part[k] = aie::add(aie::accum<accfloat, 2 * S_TILE>(
                               aie::load_v<2 * S_TILE>(t + k * 2 * S_TILE)),
                           aie::load_v<2 * S_TILE>(t + (k + P_TILES / 2) * 2 *
                                                           S_TILE))
                      .template to_vector<float>();
      AIE_LOOP_UNROLL_FULL
      for (int32_t n = P_TILES / 4; n > 0; n /= 2)
        AIE_LOOP_UNROLL_FULL
      for (int32_t k = 0; k < n; k++)
        part[k] =
            aie::add(aie::accum<accfloat, 2 * S_TILE>(part[k]), part[k + n])
                .template to_vector<float>();
      quarter[h] = fold_add(part[0].template extract<S_TILE>(0),
                            part[0].template extract<S_TILE>(1), 4);
    }
    const auto halves = fold_add(quarter[0], quarter[1], 2);
    const auto t = fold_add(halves, halves, 1);
    aie::store_v(l_new + g,
                 aie::select(aie::load_v<SM_ROWS>(l_new + g),
                             t.template extract<SM_ROWS>(0), live(g)));
  }
}
#else
// The fused block's softmax on AIE2P: S as QK^T leaves it, (4, 8) tiles in
// row-major tile order, and P as P*V reads it, (8, 8) tiles, each the two S
// tiles above one another, so a P tile is an 8-row group's 8 keys. P and the
// state are partial_softmax_rows' bit for bit, as on AIE2. S is the score
// type's, bf16 or float under MHA_F32_SCORES.
static constexpr int32_t S_TILE = 32;
static constexpr int32_t P_TILE = 64;
static constexpr int32_t BAND = 4 * ROW;
static constexpr int32_t S_TILES = ROW / 8;
static_assert(PV_R == 8 && PV_S == 8,
              "a P tile is two S tiles, one above the other");

// Out of line, as P*V's step is in attend: inlined, both spill into
// attend_block's own frame (the exponentials' constants live across QK^T's
// call), 2 KB more stack at 24x128x32, past what L1 leaves a core two key
// blocks deep.
template <typename T>
__attribute__((noinline)) static void
partial_softmax_tiles(T *__restrict S, bfloat16 *__restrict P,
                      const float *__restrict m_prev, float *__restrict m_new,
                      float *__restrict l_new, int32_t rows, int32_t valid_cols,
                      int32_t diagonal, bfloat16 scale) {
  const auto lowest =
      aie::broadcast<T, S_TILE>((T)std::numeric_limits<bfloat16>::lowest());
  auto fold_max = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::max(lo, hi);
  };
  auto fold_add = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::add(aie::accum<accfloat, decltype(a)::size()>(lo), hi)
        .template to_vector<float>();
  };
  auto live = [rows](int32_t g) {
    int32_t n = rows - g >= SM_ROWS ? SM_ROWS : rows - g > 0 ? rows - g : 0;
    return aie::mask<SM_ROWS>::from_uint32((1u << n) - 1);
  };

  // Lanes a row discards, and a padded row's every lane, hold lowest from
  // here on, as partial_softmax and band_mask leave them.
  if ((BAND_AHEAD && diagonal + AHEAD < ROW - 1) || valid_cols < ROW ||
      (BAND_BEHIND &&
       (MHA_PACKED ? 0 : (DIM_M - 1) / MHA_GROUP) + diagonal > MHA_WINDOW) ||
      rows < DIM_M) {
    // A lane's key in its tile, 0 to 7, from its index's bits.
    const auto zero = aie::zeros<int16_t, S_TILE>();
    auto key0 = aie::select(zero, aie::broadcast<int16_t, S_TILE>(1),
                            aie::mask<S_TILE>::from_uint32(0xAAAAAAAAu));
    key0 = aie::add(key0,
                    aie::select(zero, aie::broadcast<int16_t, S_TILE>(2),
                                aie::mask<S_TILE>::from_uint32(0xCCCCCCCCu)));
    key0 = aie::add(key0,
                    aie::select(zero, aie::broadcast<int16_t, S_TILE>(4),
                                aie::mask<S_TILE>::from_uint32(0xF0F0F0F0u)));
    const auto eight = aie::broadcast<int16_t, S_TILE>(8);
    for (int32_t b = 0; b < DIM_M / 4; b++) {
      int16_t end[4], start[4];
      for (int32_t r = 0; r < 4; r++) {
        const int32_t i = b * 4 + r;
        const int32_t pos = MHA_PACKED ? 0 : i / MHA_GROUP;
        int32_t e = valid_cols;
        if (BAND_AHEAD && pos + diagonal + AHEAD + 1 < e)
          e = pos + diagonal + AHEAD + 1;
        end[r] = i < rows && e > 0 ? e : 0;
        // Clamped in int32 first: a wide window's start overflows int16.
        const int32_t s = BAND_BEHIND ? pos + diagonal - MHA_WINDOW : 0;
        start[r] = s < 0 ? 0 : s > ROW ? ROW : s;
      }
      // Each against the tile's first key, which steps by 8.
      auto end_v = aie::concat(aie::broadcast<int16_t, 8>(end[0]),
                               aie::broadcast<int16_t, 8>(end[1]),
                               aie::broadcast<int16_t, 8>(end[2]),
                               aie::broadcast<int16_t, 8>(end[3]));
      auto start_v = aie::concat(aie::broadcast<int16_t, 8>(start[0]),
                                 aie::broadcast<int16_t, 8>(start[1]),
                                 aie::broadcast<int16_t, 8>(start[2]),
                                 aie::broadcast<int16_t, 8>(start[3]));
      T *__restrict t = S + b * BAND;
      for (int32_t c = 0; c < S_TILES; c++, t += S_TILE) {
        aie::mask<S_TILE> drop = aie::ge(key0, end_v);
        if constexpr (BAND_BEHIND)
          drop = drop | aie::lt(key0, start_v);
        aie::store_v(t, aie::select(aie::load_v<S_TILE>(t), lowest, drop));
        end_v = aie::sub(end_v, eight);
        start_v = aie::sub(start_v, eight);
      }
    }
  }

  // Each row's maximum, of the unscaled scores, an 8-row group (two bands)
  // at a time, and each band's over its rows' 8 lanes of an S tile; then
  // each row's eight times over, as a P tile's lanes take it.
  auto by_lane = [](aie::vector<T, SM_ROWS> m) {
    const auto [m2_lo, m2_hi] = aie::interleave_zip(m, m, 1);
    const auto m2 = aie::concat(m2_lo, m2_hi);
    const auto [m4_lo, m4_hi] = aie::interleave_zip(m2, m2, 2);
    const auto m4 = aie::concat(m4_lo, m4_hi);
    const auto [m8_lo, m8_hi] = aie::interleave_zip(m4, m4, 4);
    return aie::concat(m8_lo, m8_hi);
  };
  // Float maxima are spread again from m_new a P tile at a time: their
  // tiles would take DIM_M * 32 bytes of stack.
  constexpr bool f32 = std::is_same_v<T, float>;
  alignas(64) T m_tiles[f32 ? 1 : DIM_M * 8];
  MHA_GROUP_LOOP
  for (int32_t g = 0; g < DIM_M; g += SM_ROWS) {
    aie::accum<accfloat, SM_ROWS> prev, kept;
    prev.from_vector(aie::load_v<SM_ROWS>(m_prev + g));
    kept.from_vector(aie::load_v<SM_ROWS>(m_new + g));
    aie::vector<T, S_TILE> band_max[2];
    AIE_LOOP_UNROLL_FULL
    for (int32_t h = 0; h < 2; h++) {
      const T *__restrict t = S + (g / 4 + h) * BAND;
      auto mx = aie::load_v<2 * S_TILE>(t);
      AIE_LOOP_UNROLL_FULL
      for (int32_t c = 2; c < S_TILES; c += 2)
        mx = aie::max(mx, aie::load_v<2 * S_TILE>(t + c * S_TILE));
      band_max[h] = aie::max(mx.template extract<S_TILE>(0),
                             mx.template extract<S_TILE>(1));
    }
    const auto quads = fold_max(band_max[0], band_max[1], 4);
    const auto pairs = fold_max(quads, quads, 2);
    const aie::vector<T, SM_ROWS> o =
        fold_max(pairs, pairs, 1).template extract<SM_ROWS>(0);
    const auto m =
        aie::select(kept.to_vector<T>(),
                    aie::max(aie::max(lowest.template extract<SM_ROWS>(0), o),
                             prev.to_vector<T>()),
                    live(g));
    aie::store_v(m_new + g,
                 aie::accum<accfloat, SM_ROWS>(m).to_vector<float>());
    if constexpr (!f32)
      aie::store_v(m_tiles + g * 8, by_lane(m));
  }

  // Each P tile's exponentials, from the two S tiles it covers.
  const auto s = aie::broadcast<bfloat16, P_TILE>(scale);
  bfloat16 *__restrict w = P;
  AIE_LOOP_MIN_ITERATION_COUNT(DIM_M / 8 * S_TILES)
  for (int32_t i = 0; i < DIM_M / 8 * S_TILES; i++, w += P_TILE) {
    const T *__restrict t =
        S + (i / S_TILES) * 2 * BAND + (i % S_TILES) * S_TILE;
    const auto st =
        aie::concat(aie::load_v<S_TILE>(t), aie::load_v<S_TILE>(t + BAND));
    aie::accum<accfloat, P_TILE> x;
    if constexpr (f32 && ROW == 32) {
      // An S tile a step: 64 lanes' limbs spill, past the stack L1 leaves a
      // 24x128x32 block two key blocks deep.
      const auto m =
          by_lane(aie::load_v<SM_ROWS>(m_new + (i / S_TILES) * SM_ROWS));
      const auto s32 = s.template extract<S_TILE>(0);
      AIE_LOOP_UNROLL_FULL
      for (int32_t h = 0; h < 2; h++) {
        aie::accum<accfloat, S_TILE> d;
        d.from_vector(aie::load_v<S_TILE>(t + h * BAND));
        const Limbs<S_TILE> dl =
            bf16_limbs(aie::sub(d, m.template extract<S_TILE>(h)));
        aie::accum<accfloat, S_TILE> y = aie::mul(dl.hi, s32);
        y = aie::mac(y, dl.mid, s32);
        y = aie::mac(y, dl.lo, s32);
        aie::store_v(w + h * S_TILE, exp2_bf16(y.to_vector<float>()));
      }
      continue;
    } else if constexpr (f32) {
      // As partial_softmax_rows: S - m in float32, times s as three limbs.
      aie::accum<accfloat, P_TILE> d;
      d.from_vector(st);
      const Limbs<P_TILE> dl = bf16_limbs(aie::sub(
          d, by_lane(aie::load_v<SM_ROWS>(m_new + (i / S_TILES) * SM_ROWS))));
      x = aie::mul(dl.hi, s);
      x = aie::mac(x, dl.mid, s);
      x = aie::mac(x, dl.lo, s);
    } else {
      x = aie::msc(aie::mul(st, s),
                   aie::load_v<P_TILE>(m_tiles + (i / S_TILES) * P_TILE), s);
    }
    aie::store_v(w, exp2_bf16(x.to_vector<float>()));
  }

  // Each row's sum over its P tiles, eight rows at a time, a P tile's upper
  // and lower four apart: tiles half the row apart, then half that, to one
  // tile, then lanes four, two and one apart.
  MHA_GROUP_LOOP
  for (int32_t g = 0; g < DIM_M; g += SM_ROWS) {
    aie::vector<float, S_TILE> half[2];
    AIE_LOOP_UNROLL_FULL
    for (int32_t h = 0; h < 2; h++) {
      const bfloat16 *__restrict t = P + g * ROW + h * S_TILE;
      aie::vector<float, S_TILE> part[S_TILES / 2];
      AIE_LOOP_UNROLL_FULL
      for (int32_t k = 0; k < S_TILES / 2; k++)
        part[k] = aie::add(aie::accum<accfloat, S_TILE>(
                               aie::load_v<S_TILE>(t + k * P_TILE)),
                           aie::load_v<S_TILE>(t + (k + S_TILES / 2) * P_TILE))
                      .template to_vector<float>();
      AIE_LOOP_UNROLL_FULL
      for (int32_t n = S_TILES / 4; n > 0; n /= 2)
        AIE_LOOP_UNROLL_FULL
      for (int32_t k = 0; k < n; k++)
        part[k] = aie::add(aie::accum<accfloat, S_TILE>(part[k]), part[k + n])
                      .template to_vector<float>();
      half[h] = part[0];
    }
    const auto quads = fold_add(half[0], half[1], 4);
    const auto pairs = fold_add(quads.template extract<S_TILE / 2>(0),
                                quads.template extract<S_TILE / 2>(1), 2);
    const auto t = fold_add(pairs.template extract<SM_ROWS>(0),
                            pairs.template extract<SM_ROWS>(1), 1);
    aie::store_v(l_new + g,
                 aie::select(aie::load_v<SM_ROWS>(l_new + g), t, live(g)));
  }
}
#endif

// The fused block's K and V come as one stream: each key's K row, then its
// V row, in QK^T's B tiling, (KV_KEYS, 8) tiles of KV_KEYS keys by 8
// columns, which on AIE2P is also P*V's V tiling. A group of KV_KEYS keys
// is its DIM_K / 8 K tiles, then its V tiles; with MHA_KV_HEADS, each of
// that many heads' in turn, a position's heads one run in a cache.
#ifndef MHA_KV_HEADS
#define MHA_KV_HEADS 1
#endif
static constexpr int32_t KV_KEYS = AIE_ARCH_AIE2 ? 4 : 8;
static constexpr int32_t KV_TILE = KV_KEYS * 8;
static constexpr int32_t KV_PITCH = MHA_KV_HEADS * 2 * DIM_K / 8;
static constexpr int32_t KV_ROW = KV_PITCH * KV_TILE;

#if AIE_ARCH_AIE2
// GEMM's P V, its (8, 4) B tile of 8 keys by 4 columns half the columns of
// two such V tiles, which one unzip takes apart; each output tile takes the
// same products in the same order as matmul_rowmaj's.
template <unsigned M, unsigned K, unsigned N>
static void matmul_pv_kv(const bfloat16 *__restrict pA,
                         const bfloat16 *__restrict pV, float *__restrict pC) {
  constexpr unsigned r = 4, s = 8, t = 4;
  constexpr unsigned R = (M / r) % 4 == 0 ? 4 : 2;
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
  constexpr unsigned rowA = M / r, colA = K / s, colB = N / t;
  static_assert(rowA % R == 0 && colB % 4 == 0);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += R) {
    // GEMM's A cursor, wrapping to its row block's start, keeps A's loads
    // in the k loop rather than hoisted over j and spilled.
    const bfloat16 *pAk = pA + z * colA * MMUL::size_A;
    dims_2d_t a_dims =
        dims_2d_from_steps(colA, MMUL::size_A * sizeof(bfloat16), 0);
    for (unsigned j = 0; j < colB; j += 4) {
      // A pointer a row block, as GEMM's, so each tile is a short offset.
      float *__restrict pCx[R];
      MMUL C[R][4];
      AIE_LOOP_UNROLL_FULL
      for (unsigned x = 0; x < R; x++) {
        pCx[x] = pC + ((z + x) * colB + j) * MMUL::size_C;
        AIE_LOOP_UNROLL_FULL
        for (unsigned y = 0; y < 4; y++)
          C[x][y] = MMUL(aie::load_v<MMUL::size_C>(pCx[x] + y * MMUL::size_C));
      }
      const bfloat16 *__restrict pVk = pV + (j / 2) * KV_TILE;
      AIE_LOOP_UNROLL_FULL
      for (unsigned i = 0; i < colA; i++) {
        aie::vector<bfloat16, MMUL::size_A> A[R];
        AIE_LOOP_UNROLL_FULL
        for (unsigned x = 0; x < R; x++)
          A[x] = aie::load_v<MMUL::size_A>(pAk + x * colA * MMUL::size_A);
        pAk = add_2d_byte(pAk, a_dims);
        const auto [B0, B1] = aie::interleave_unzip(
            aie::load_v<KV_TILE>(pVk), aie::load_v<KV_TILE>(pVk + KV_ROW), 4);
        const auto [B2, B3] = aie::interleave_unzip(
            aie::load_v<KV_TILE>(pVk + KV_TILE),
            aie::load_v<KV_TILE>(pVk + KV_TILE + KV_ROW), 4);
        pVk += 2 * KV_ROW;
        AIE_LOOP_UNROLL_FULL
        for (unsigned x = 0; x < R; x++) {
          C[x][0].mac(A[x], B0);
          C[x][1].mac(A[x], B1);
          C[x][2].mac(A[x], B2);
          C[x][3].mac(A[x], B3);
        }
      }
      AIE_LOOP_UNROLL_FULL
      for (unsigned x = 0; x < R; x++)
        AIE_LOOP_UNROLL_FULL
      for (unsigned y = 0; y < 4; y++)
        aie::store_v(pCx[x] + y * MMUL::size_C,
                     C[x][y].template to_vector<float>());
    }
  }
}

// zero_rows_from for V in the fused block's tiling: a key's 8 columns of a
// tile are its 8 lanes.
static void zero_kv_rows_from(bfloat16 *V, int32_t row) {
  const auto zeros = aie::zeros<bfloat16, KV_TILE>();
  bfloat16 *__restrict band = V + (row / 4) * KV_ROW;
  if (row % 4) {
    const auto stale =
        aie::mask<KV_TILE>::from_uint32(~((1u << (8 * (row % 4))) - 1));
    bfloat16 *__restrict p = band;
    for (int32_t j = 0; j < DIM_K / 8; j++, p += KV_TILE)
      aie::store_v(p, aie::select(aie::load_v<KV_TILE>(p), zeros, stale));
    band += KV_ROW;
  }
  for (; band < V + DIM_N / 4 * KV_ROW; band += KV_ROW) {
    bfloat16 *__restrict p = band;
    for (int32_t j = 0; j < DIM_K / 8; j++, p += KV_TILE)
      aie::store_v(p, zeros);
  }
}
#else
// matmul_rowmaj's P V, V's key groups KV_ROW apart.
template <unsigned M, unsigned K, unsigned N>
static void matmul_pv_kv(const bfloat16 *__restrict pA,
                         const bfloat16 *__restrict pV, float *__restrict pC) {
  mm_bf16_2x2_aie2p<M / PV_R, K / PV_S, N / PV_T, PV_R, PV_S, PV_T, false, true,
                    KV_PITCH>(pA, pV, pC);
}

// zero_rows_from for V in the fused block's tiling.
static void zero_kv_rows_from(bfloat16 *V, int32_t row) {
  const auto zeros = aie::zeros<bfloat16, KV_TILE>();
  bfloat16 *__restrict band = V + (row / KV_KEYS) * KV_ROW;
  if (row % KV_KEYS) {
    const auto stale =
        aie::mask<KV_TILE>::from_uint64(~uint64_t(0) << (row % KV_KEYS) * 8);
    bfloat16 *__restrict p = band;
    for (int32_t j = 0; j < DIM_K / 8; j++, p += KV_TILE)
      aie::store_v(p, aie::select(aie::load_v<KV_TILE>(p), zeros, stale));
    band += KV_ROW;
  }
  for (; band < V + DIM_N / KV_KEYS * KV_ROW; band += KV_ROW) {
    bfloat16 *__restrict p = band;
    for (int32_t j = 0; j < DIM_K / 8; j++, p += KV_TILE)
      aie::store_v(p, zeros);
  }
}
#endif
#endif

// V's rows from row on are keys past S_kv, which do not exist: their weights
// in P are 0, but the product still reads the rows, whatever the buffer held.
// Emulated through bfp16 it shares one exponent over eight rows of the
// reduction, so a stale row changes the valid rows beside it, and 0 * NaN is
// NaN. V is DIM_N keys of the head dimension DIM_K, blocked as matmul_PV reads
// it: (PV_S, PV_T) tiles, row-major, and a band of DIM_K / PV_T tiles is PV_S
// rows.
static void zero_rows_from(bfloat16 *V, int32_t row) {
  constexpr int32_t tile = PV_S * PV_T;
  constexpr int32_t band = DIM_K / PV_T;
  const auto zeros = aie::zeros<bfloat16, tile>();
  bfloat16 *__restrict p = V + (row / PV_S) * band * tile;
  if (row % PV_S) {
    const auto stale =
        aie::ge(aie::load_v<tile>(sm_lane_idx),
                aie::broadcast<int16_t, tile>((row % PV_S) * PV_T));
    for (int32_t j = 0; j < band; j++, p += tile) {
      aie::store_v(p, aie::select(aie::load_v<tile>(p), zeros, stale));
    }
  }
  for (; p < V + DIM_N * DIM_K; p += tile) {
    aie::store_v(p, zeros);
  }
}

// matmul_PV's step: O, rescaled by the state's factor past the first key
// block, plus P V; with kv_tiled, V in K's tiling as attend_block takes it.
template <bool kv_tiled>
static inline __attribute__((always_inline)) void
pv_accumulate(bfloat16 *P, bfloat16 *V, float *out, float *scale_buffer,
              const int32_t B_q, int32_t first_iter, int32_t *idx_buffer,
              int32_t S_kv_eff) {
  if (first_iter == 0) {
    zero_vectorized<float, DIM_M, DIM_K, false>(out);
  }

  // Skip a key block no row of the query block keeps, as partial_softmax does.
  if (!block_live(block_diagonal(idx_buffer, B_q, DIM_N, S_kv_eff), B_q,
                  DIM_N)) {
    return;
  }

  int32_t valid_kv_rows = S_kv_eff - idx_buffer[0] * DIM_N;
  // A block past every key: partial_softmax left the state, and its rescale
  // factor is the previous block's.
  if (valid_kv_rows <= 0) {
    return;
  }
  if (valid_kv_rows < DIM_N) {
#ifdef MHA_FUSED
    if constexpr (kv_tiled)
      zero_kv_rows_from(V, valid_kv_rows);
    else
#endif
      zero_rows_from(V, valid_kv_rows);
  }

  // O_{i-1} is scaled by exp2((m_{i-1} - m_i) s); on the first block O is
  // zero.
  if (first_iter != 0) {
    rescale_running_O(out, scale_buffer + 3 * B_q, B_q);
  }
#ifdef MHA_FUSED
  if constexpr (kv_tiled)
    matmul_pv_kv<DIM_M, DIM_N, DIM_K>(P, V, out);
  else
#endif
    matmul_rowmaj<float, DIM_M, DIM_N, DIM_K>(P, V, out);
}

// The online state is four B_q-long regions, B_q the build's DIM_M, walked at
// the widest of 64 (32 on AIE2, where 64 float lanes cost program memory), 32
// and 16 lanes that divides it, or 8 when DIM_M is no multiple of 16.
static constexpr int STATE_W_MAX = AIE_ARCH_AIE2 ? 32 : VECTOR_LENGTH;
static constexpr int STATE_W = DIM_M % 16                 ? 8
                               : DIM_M % STATE_W_MAX == 0 ? STATE_W_MAX
                               : DIM_M % 32 == 0          ? 32
                                                          : 16;

// The state is float: l and O gain a key block's worth at a time, and in bf16
// a long context's later blocks round away against what came before.
// The maxima are unscaled scores, each a bf16: their difference is scaled by
// the exact products the exponentials take.
// Out of line: inlined into partial_softmax, it slows the row loops by about
// 270 cycles a 64x64 block on npu1.
template <int W>
__attribute__((noinline)) static void
advance_state_w(float *scale_buffer, int32_t B_q, bfloat16 scale) {
  using VecW = aie::vector<float, W>;
  const auto s = aie::broadcast<bfloat16, W>(scale);
  for (int32_t i = 0; i < B_q; i += W) {
    VecW m_i = aie::load_v<W>(scale_buffer + B_q + i);
    VecW l_i_minus_1 = aie::load_v<W>(scale_buffer + 2 * B_q + i);
    VecW accum_exp_val = aie::load_v<W>(scale_buffer + 3 * B_q + i);
    const VecW m_prev = aie::load_v<W>(scale_buffer + i);
    // Rows whose maxima all held: a factor of exp2(0) = 1, and l adds the
    // block's sum, as below, without the exponential.
    if (aie::equal(m_i, m_prev)) {
      aie::store_v(scale_buffer + 3 * B_q + i, aie::broadcast<float, W>(1.0f));
      aie::store_v(scale_buffer + 2 * B_q + i,
                   aie::add(aie::accum<accfloat, W>(accum_exp_val), l_i_minus_1)
                       .template to_vector<float>());
      continue;
    }

    aie::accum<accfloat, W> prev;
    prev.from_vector(m_prev);
#ifdef MHA_F32_SCORES
    // Float maxima; see partial_softmax_rows.
    const Limbs<W> dl = bf16_limbs(aie::sub(prev, m_i));
    aie::accum<accfloat, W> diff = aie::mul(dl.hi, s);
    diff = aie::mac(diff, dl.mid, s);
    diff = aie::mac(diff, dl.lo, s);
#else
    aie::accum<accfloat, W> cur;
    cur.from_vector(m_i);
    aie::accum<accfloat, W> diff =
        aie::msc(aie::mul(prev.template to_vector<bfloat16>(), s),
                 cur.template to_vector<bfloat16>(), s);
#endif
    const auto max_diff_exp = exp2_bf16(diff.template to_vector<float>());
    const VecW factor =
        aie::accum<accfloat, W>(max_diff_exp).template to_vector<float>();

    aie::store_v(scale_buffer + 3 * B_q + i, factor);
    aie::accum<accfloat, W> l_prev;
    l_prev.from_vector(l_i_minus_1);
    const Limbs<W> ll = bf16_limbs(l_prev);
    aie::accum<accfloat, W> l_i;
    l_i.from_vector(accum_exp_val);
    l_i = aie::mac(l_i, ll.hi, max_diff_exp);
    l_i = aie::mac(l_i, ll.mid, max_diff_exp);
    l_i = aie::mac(l_i, ll.lo, max_diff_exp);
    // A row whose max held adds its block's sum to l once, as O keeps its
    // rows: a key block that keeps none of a row's keys then leaves it as it
    // was, wherever the row's query block starts.
    const aie::accum<accfloat, W> l_held =
        aie::add(aie::accum<accfloat, W>(accum_exp_val), l_i_minus_1);
    aie::store_v(scale_buffer + 2 * B_q + i,
                 aie::select(l_i.template to_vector<float>(),
                             l_held.template to_vector<float>(),
                             aie::eq(factor, aie::broadcast<float, W>(1.0f))));
    aie::store_v(scale_buffer + i, m_i);
  }
}

template <int W>
static inline void init_state_w(float *scale_buffer, int32_t size) {
  using VecW = aie::vector<float, W>;
  VecW lowest_vec =
      aie::broadcast<float, W>((float)std::numeric_limits<bfloat16>::lowest());
  VecW zeros_vec = aie::zeros<float, W>();
  for (int32_t i = 0; i < size; i += W) {
    // VJUNG: m_{i-1} vector
    aie::store_v(scale_buffer + i, lowest_vec);
    // VJUNG: m_{i} vector
    aie::store_v(scale_buffer + size + i, zeros_vec);
    // VJUNG: l_{i} vector
    aie::store_v(scale_buffer + 2 * size + i, zeros_vec);
  }
}

template <int W>
static inline void invert_sums_w(float *scale_buffer, int32_t B_q) {
  // A row past S_q attended over nothing: its sum is 0 and its O row is 0,
  // and 0 * (1 / 0) would make it NaN. It is scaled by 0 instead. Every other
  // row keeps a key, so its sum is positive.
  const auto zeros = aie::zeros<float, W>();
  for (int32_t i = 0; i < B_q; i += W) {
    aie::vector<float, W> l_vec = aie::load_v<W>(scale_buffer + 2 * B_q + i);
    aie::mask<W> empty = aie::eq(l_vec, zeros);
    l_vec = aie::select(aie::inv(l_vec), zeros, empty);
    aie::store_v(scale_buffer + 2 * B_q + i, l_vec);
  }
}

extern "C" {
void passThroughLine(int32_t *in, int32_t *out, int32_t lineWidth);

void matmul_bf16_bf16_wrapper(bfloat16 *a_in, bfloat16 *b_in, score_t *c_out,
                              int32_t *idx_buffer) {
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  // Packed, the rows' position is S_kv_eff's, which this call does not take,
  // so every block is computed.
  if (!MHA_PACKED &&
      !block_live(block_diagonal(idx_buffer, DIM_M, DIM_N, 0), DIM_M, DIM_N)) {
    // A product below is one interval; an empty pair keeps a masked call to
    // one interval too, so every call can be timed.
    event0();
    event1();
    ::aie::set_rounding(saved_rounding);
    return;
  }

#if AIE_ARCH_AIE2
  event0();
  matmul_aie2<score_t, DIM_M, DIM_K, DIM_N, is_b_row_maj>(a_in, b_in, c_out);
  event1();
#elif defined(MHA_F32_SCORES)
  matmul_bf16_f32(a_in, b_in, c_out);
#elif defined(MHA_QK_UNROLL_K)
  matmul_vectorized_2x2_mmul<bfloat16, bfloat16, DIM_M / 4, DIM_K / 8,
                             DIM_N / 8, 4, 8, 8, is_b_row_maj, is_c_row_maj,
                             true>(a_in, b_in, c_out);
#else
  matmul_bf16_bf16(a_in, b_in, c_out);
#endif
  ::aie::set_rounding(saved_rounding);
}

void matmul_bf16_bf16_wrapper_scalar(bfloat16 *a_in, bfloat16 *b_in,
                                     score_t *c_out) {
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);
#ifdef MHA_F32_SCORES
  matmul_scalar_bf16_f32(a_in, b_in, c_out);
#else
  matmul_scalar_bf16_bf16(a_in, b_in, c_out);
#endif
  ::aie::set_rounding(saved_rounding);
}

// out is the float accumulator O, which the first key block starts: P, a
// B_q by DIM_N block of weights, times V, DIM_N keys by the head dimension
// DIM_K.
void matmul_PV(bfloat16 *Q, bfloat16 *K, float *out, float *scale_buffer,
               const int32_t B_q, int32_t first_iter, int32_t *idx_buffer,
               int32_t S_kv_eff) {
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);
  pv_accumulate<false>(Q, K, out, scale_buffer, B_q, first_iter, idx_buffer,
                       S_kv_eff);
  ::aie::set_rounding(saved_rounding);
  event1();
}

// O, the float accumulator, divided by the row sums into out, which may be
// O itself: each band's bf16 lands on bytes of O already read.
void rescale_O(float *O, bfloat16 *out, float *scale_buffer, int32_t B_q,
               int32_t *idx_buffer) {
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  // Once per query block, so one width: a float inv is long code on aie2.
  invert_sums_w<DIM_M % 16 ? 8 : 16>(scale_buffer, B_q);

  // VJUNG: Only after all KV are processed
  // VJUNG: TODO: Make this generic for every tile size
  // VJUNG: Need to scale depending on the data layout at the output of GEMM
  // VJUNG: Scale O_{i} by 1/l_{i}
  scale_blocked_rows(O, out, scale_buffer + 2 * B_q, B_q);
  ::aie::set_rounding(saved_rounding);
  event1();
}

void partial_softmax(score_t *A, bfloat16 *P, float *scale_buffer,
                     int32_t *idx_buffer, bfloat16 inv_scale, int32_t B_q,
                     int32_t B_kv, int32_t S_q_eff, int32_t S_kv_eff) {
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  // Block indices
  int32_t q_block_idx = idx_buffer[1];
  int32_t kv_block_idx = idx_buffer[0];

  // The band in global positions: query row i keeps the key columns from
  // i + diagonal - MHA_WINDOW to i + diagonal + AHEAD, so a query block
  // narrower than the key block masks at the right column. Skip a block no
  // row keeps a key of.
  const int32_t diagonal = block_diagonal(idx_buffer, B_q, B_kv, S_kv_eff);
  if (!block_live(diagonal, B_q, B_kv)) {
    zero_vectorized<bfloat16, DIM_M, DIM_N>(P);
    ::aie::set_rounding(saved_rounding);
    return;
  }

  // Compute valid extents within this block for padded tails
  int32_t valid_q_rows = S_q_eff - q_block_idx * B_q;
  if (valid_q_rows < 0)
    valid_q_rows = 0;
  if (valid_q_rows > B_q)
    valid_q_rows = B_q;

  int32_t valid_kv_cols = S_kv_eff - kv_block_idx * B_kv;
  if (valid_kv_cols < 0)
    valid_kv_cols = 0;
  if (valid_kv_cols > B_kv)
    valid_kv_cols = B_kv;

  // Fully padded block: contributes nothing
  if (valid_q_rows == 0 || valid_kv_cols == 0) {
    zero_vectorized<bfloat16, DIM_M, DIM_N>(P);
    ::aie::set_rounding(saved_rounding);
    return;
  }

  // Not at entry: zero_vectorized brackets itself, so a skipped block is
  // timed by its own pair and every call still emits exactly one.
  event0();

  const auto lowest = aie::broadcast<score_t, ROW>(
      (score_t)std::numeric_limits<bfloat16>::lowest());

  // Tail mask: invalidate padded Q rows.  They are the end of A, so the walk
  // over (row, vector) is one linear fill.
  for (int32_t n = valid_q_rows * ROW; n < B_q * ROW; n += ROW) {
    aie::store_v(A + n, lowest);
  }

  // A row's maximum is taken before scaling, so inv_scale must be positive
  // and finite, as every caller's 1/sqrt(d) is.
  partial_softmax_rows(A, P, scale_buffer, scale_buffer + B_q,
                       scale_buffer + 3 * B_q, valid_q_rows, valid_kv_cols,
                       diagonal, inv_scale);
  // A row whose band starts past this block's keys has kept no key yet, the
  // key blocks coming in order: its maximum is still lowest, so every lane
  // came to 2^0, which only its first key's rescale, by 2^(lowest * s), would
  // take back off. It adds nothing instead, its sum 0 and its P row zeroed
  // with the padded rows'. Such rows are a suffix.
  int32_t kept_rows = valid_q_rows;
  if constexpr (BAND_BEHIND && !MHA_PACKED) {
    const int32_t dead = valid_kv_cols + MHA_WINDOW - diagonal;
    for (; kept_rows > dead * MHA_GROUP && kept_rows > 0; kept_rows--)
      scale_buffer[3 * B_q + kept_rows - 1] = 0.0f;
  }
  // Zero out P rows corresponding to padded Q rows, which are again a suffix
  // and so one linear fill.
  for (int32_t n = kept_rows * ROW; n < B_q * ROW; n += ROW) {
    aie::store_v(P + n, aie::zeros<bfloat16, ROW>());
  }

  advance_state_w<STATE_W>(scale_buffer, B_q, inv_scale);
  ::aie::set_rounding(saved_rounding);
  event1();
}

#ifdef MHA_FUSED
// Out of line: inlined into attend_block at 16x256x16, Peano fails to
// legalize a G_BITCAST of <16 x s16>.
__attribute__((noinline)) static void qk_kv(const bfloat16 *Q,
                                            const bfloat16 *KV, score_t *S) {
#if AIE_ARCH_AIE2
  matmul_aie2<score_t, DIM_M, DIM_K, DIM_N, is_b_row_maj, KV_PITCH, true>(Q, KV,
                                                                          S);
#else
  // Rolled, unlike matmul_bf16_bf16_wrapper's at DIM_K 64: starting C at 0
  // unrolled spills 8.5 KB of stack and ran 572 cycles slower a 32x64x64
  // block.
  matmul_vectorized_2x2_mmul<bfloat16, score_t, DIM_M / 4, DIM_K / 8, DIM_N / 8,
                             4, 8, 8, is_b_row_maj, true, false, KV_PITCH, true,
                             false>(Q, KV, S);
#endif
}

#ifdef MHA_REL_BAND
// The relative band (Gemma 4 audio's relative-position logits): score (i, j)
// gains Q_i . R_d at distance d = pos_i - key_j, 0 <= d < REL_KEYS, R's rows
// past the model's window zero. relative_rows takes QR = Q R^T once a query
// block, each row reversed into REL_ROW floats, d at REL_AT - d with zeros
// either side, so a tile row's four addends are one run of them.
static constexpr int32_t REL_KEYS = 16, REL_ROW = 32, REL_AT = 23;
static_assert(MHA_GROUP == 1 && !MHA_PACKED,
              "the relative band takes a query a row");

// Before the cap, as a pre-cap addend: each S tile within the band.
static inline __attribute__((always_inline)) void
relative_add(float *S, const float *QR, int32_t diagonal) {
  for (int32_t b = 0; b < DIM_M / 4; b++) {
    for (int32_t c = 0; c < S_TILES; c++) {
      // Lane (r, k) of the tile is at distance d0 + r - k.
      const int32_t d0 = diagonal + 4 * b - 4 * c;
      if (d0 + 3 < 0 || d0 - 3 >= REL_KEYS)
        continue;
      // Each row's run of eight, the first four its, two rows a vector.
      aie::vector<float, 8> rows[4];
      for (int32_t r = 0; r < 4; r++) {
        const int32_t d = d0 + r;
        rows[r] = d < 0 || d - 3 >= REL_KEYS
                      ? aie::zeros<float, 8>()
                      : aie::load_unaligned_v<8>(QR + (4 * b + r) * REL_ROW +
                                                 REL_AT - d);
      }
      float *t = S + (b * S_TILES + c) * S_TILE;
      aie::store_v(
          t, aie::add(aie::accum<accfloat, S_TILE>(aie::load_v<S_TILE>(t)),
                      aie::concat(
                          aie::filter_even(aie::concat(rows[0], rows[1]), 4),
                          aie::filter_even(aie::concat(rows[2], rows[3]), 4)))
                 .template to_vector<float>());
    }
  }
}
#endif

extern "C++" { // attend is a template

// One key block of a query block on one core: S = Q K^T (scores(S) writes
// it), the online softmax over S, then O += P V, bit for bit as
// matmul_bf16_bf16_wrapper, partial_softmax and matmul_PV compute it over
// three cores. KV is the block's K, then its V, both in QK^T's B tiling, so
// one stream carries them; S holds its scores and P its weights.
// init_scale_buffer starts a query block, rescale_O ends it. QR is
// relative_rows' (MHA_REL_BAND).
template <typename Scores>
static inline __attribute__((always_inline)) void
attend(Scores scores, bfloat16 *KV, const float *QR, float *O, score_t *S,
       bfloat16 *P, float *scale_buffer, int32_t *idx_buffer,
       bfloat16 inv_scale, int32_t S_q_eff, int32_t S_kv_eff) {
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);
  constexpr int32_t B_q = DIM_M;
  static_assert(DIM_M % MHA_GROUP == 0, "a block holds whole positions");
  bfloat16 *V = KV + DIM_K / 8 * KV_TILE;
  const int32_t first_iter = idx_buffer[0];
  const int32_t diagonal = block_diagonal(idx_buffer, B_q, DIM_N, S_kv_eff);
  int32_t valid_kv_cols = S_kv_eff - idx_buffer[0] * DIM_N;
  int32_t valid_q_rows = S_q_eff - idx_buffer[1] * B_q;
  // A block no row keeps a key of, past every key or every query: the
  // three-core steps leave the state and O as they were, but for the first
  // block's zero O, which a block of no queries adds only +0 to.
  if (!block_live(diagonal, B_q, DIM_N) || valid_kv_cols <= 0 ||
      valid_q_rows <= 0) {
    if (first_iter == 0)
      zero_vectorized<float, DIM_M, DIM_K, false>(O);
    ::aie::set_rounding(saved_rounding);
    event1();
    return;
  }
  if (valid_kv_cols > DIM_N)
    valid_kv_cols = DIM_N;
  if (valid_q_rows > B_q)
    valid_q_rows = B_q;
  scores(S);
#ifdef MHA_REL_BAND
  relative_add(S, QR, diagonal);
#endif
#ifdef MHA_SOFT_CAP
  // Each score cap * tanh(s / cap) * log2(e): the scale the caller passes is
  // then 1, which the limbs of m - S take exactly.
  static_assert(std::is_same_v<score_t, float>, "a soft cap takes f32 scores");
  for (int32_t n = 0; n < DIM_M * DIM_N; n += 32)
    aie::store_v(
        S + n, soft_cap_log2e(aie::accum<accfloat, 32>(aie::load_v<32>(S + n)),
                              MHA_SOFT_CAP)
                   .to_vector<float>());
#endif
  partial_softmax_tiles(S, P, scale_buffer, scale_buffer + B_q,
                        scale_buffer + 3 * B_q, valid_q_rows, valid_kv_cols,
                        diagonal, inv_scale);
  // As partial_softmax: a dead suffix of rows adds nothing, and P's rows
  // from the first padded or dead one on are zero.
  int32_t kept_rows = valid_q_rows;
  if constexpr (BAND_BEHIND && !MHA_PACKED) {
    const int32_t dead = valid_kv_cols + MHA_WINDOW - diagonal;
    for (; kept_rows > dead * MHA_GROUP && kept_rows > 0; kept_rows--)
      scale_buffer[3 * B_q + kept_rows - 1] = 0.0f;
  }
  if (kept_rows < B_q) {
    // An index, not a pointer carried from one loop to the next: Peano
    // fails to select a G_SEXT of the latter's 20-bit modifier.
#if AIE_ARCH_AIE2
    int32_t n = (kept_rows / 4) * BAND;
    if (kept_rows % 4) {
      const auto keep =
          aie::mask<2 * S_TILE>::from_uint32((1u << (8 * (kept_rows % 4))) - 1);
      for (int32_t c = 0; c < P_TILES; c++, n += 2 * S_TILE)
        aie::store_v(P + n, aie::select(aie::zeros<bfloat16, 2 * S_TILE>(),
                                        aie::load_v<2 * S_TILE>(P + n), keep));
    }
    for (; n < DIM_M * DIM_N; n += 2 * S_TILE)
      aie::store_v(P + n, aie::zeros<bfloat16, 2 * S_TILE>());
#else
    int32_t n = (kept_rows / 8) * 2 * BAND;
    if (kept_rows % 8) {
      const auto keep = aie::mask<P_TILE>::from_uint64(
          (uint64_t(1) << (8 * (kept_rows % 8))) - 1);
      for (int32_t c = 0; c < S_TILES; c++, n += P_TILE)
        aie::store_v(P + n, aie::select(aie::zeros<bfloat16, P_TILE>(),
                                        aie::load_v<P_TILE>(P + n), keep));
    }
    for (; n < DIM_M * DIM_N; n += P_TILE)
      aie::store_v(P + n, aie::zeros<bfloat16, P_TILE>());
#endif
  }
  advance_state_w<STATE_W>(scale_buffer, B_q, inv_scale);
#if AIE_ARCH_AIE2P
  [[clang::noinline]]
#endif
  pv_accumulate<true>(P, V, O, scale_buffer, B_q, first_iter, idx_buffer,
                      S_kv_eff);
  ::aie::set_rounding(saved_rounding);
  event1();
}
} // extern "C++"

#if defined(MHA_PAIR)
#ifdef MHA_REL_BAND
#error "the relative band's QR takes a whole query row, not a pair's half"
#endif
// The head dimension split over two neighbouring cores, DIM_K of it each:
// pair_scores writes a core's half of Q K^T in float32 for the other core to
// read, and attend_pair adds the two halves' partials, S0 the first's and S1
// the second's in that order on both cores, so both hold one S and one P;
// each then accumulates its half of O from its half of V.
void pair_scores(const bfloat16 *Q, const bfloat16 *KV, float *S_part,
                 int32_t *idx_buffer, int32_t S_q_eff, int32_t S_kv_eff) {
  event0();
  // A block attend skips needs no scores.
  if (block_live(block_diagonal(idx_buffer, DIM_M, DIM_N, S_kv_eff), DIM_M,
                 DIM_N) &&
      S_kv_eff > idx_buffer[0] * DIM_N && S_q_eff > idx_buffer[1] * DIM_M)
    matmul_aie2<float, DIM_M, DIM_K, DIM_N, is_b_row_maj, KV_PITCH, true>(
        Q, KV, S_part);
  event1();
}

void attend_pair(const float *S0, const float *S1, bfloat16 *KV, float *O,
                 score_t *S, bfloat16 *P, float *scale_buffer,
                 int32_t *idx_buffer, bfloat16 inv_scale, int32_t S_q_eff,
                 int32_t S_kv_eff) {
  attend(
      [S0, S1](score_t *S) {
        AIE_LOOP_UNROLL_FULL
        for (int32_t n = 0; n < DIM_M * DIM_N; n += 32)
          aie::store_v(
              S + n, aie::add(aie::accum<accfloat, 32>(aie::load_v<32>(S0 + n)),
                              aie::load_v<32>(S1 + n))
                         .template to_vector<score_t>());
      },
      KV, nullptr, O, S, P, scale_buffer, idx_buffer, inv_scale, S_q_eff,
      S_kv_eff);
}
#elif defined(MHA_REL_BAND)
void attend_block(bfloat16 *Q, bfloat16 *KV, const float *QR, float *O,
                  score_t *S, bfloat16 *P, float *scale_buffer,
                  int32_t *idx_buffer, bfloat16 inv_scale, int32_t S_q_eff,
                  int32_t S_kv_eff) {
  attend([Q, KV](score_t *S) { qk_kv(Q, KV, S); }, KV, QR, O, S, P,
         scale_buffer, idx_buffer, inv_scale, S_q_eff, S_kv_eff);
}

// QR = Q R^T, laid out as relative_add reads it. R is a key block's object
// whose first REL_KEYS keys' K rows are the head's relative keys, so they
// ride the KV stream. The product lands in QR's second half: a row group's
// rows overwrite its tiles only after reading them.
void relative_rows(const bfloat16 *Q, const bfloat16 *R, float *QR) {
  static_assert(DIM_N >= REL_KEYS, "the relative keys are a key block's");
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);
  float *c = QR + DIM_M * REL_KEYS;
  // At 16-key blocks this is attend_block's own QK^T: one copy of it.
  if constexpr (DIM_N == REL_KEYS)
    qk_kv(Q, R, c);
  else
    matmul_aie2<float, DIM_M, DIM_K, REL_KEYS, is_b_row_maj, KV_PITCH, true>(
        Q, R, c);
  for (int32_t b = 0; b < DIM_M / 4; b++) {
    aie::vector<float, S_TILE> t[4];
    for (int32_t k = 0; k < 4; k++)
      t[k] = aie::load_v<S_TILE>(c + (4 * b + k) * S_TILE);
    for (int32_t r = 0; r < 4; r++) {
      const auto row =
          aie::concat(t[0].template extract<4>(r), t[1].template extract<4>(r),
                      t[2].template extract<4>(r), t[3].template extract<4>(r));
      float *dst = QR + (4 * b + r) * REL_ROW;
      aie::store_v(dst, aie::zeros<float, 8>());
      aie::store_v(dst + REL_AT - (REL_KEYS - 1), aie::reverse(row));
      aie::store_v(dst + REL_AT + 1, aie::zeros<float, 8>());
    }
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}
#else
void attend_block(bfloat16 *Q, bfloat16 *KV, float *O, score_t *S, bfloat16 *P,
                  float *scale_buffer, int32_t *idx_buffer, bfloat16 inv_scale,
                  int32_t S_q_eff, int32_t S_kv_eff) {
  attend([Q, KV](score_t *S) { qk_kv(Q, KV, S); }, KV, nullptr, O, S, P,
         scale_buffer, idx_buffer, inv_scale, S_q_eff, S_kv_eff);
}

// attend_block on head `head` of a key block's MHA_KV_HEADS heads.
void attend_head(bfloat16 *Q, bfloat16 *KV, float *O, score_t *S, bfloat16 *P,
                 float *scale_buffer, int32_t *idx_buffer, bfloat16 inv_scale,
                 int32_t S_q_eff, int32_t S_kv_eff, int32_t head) {
  attend_block(Q, KV + head * 2 * KV_KEYS * DIM_K, O, S, P, scale_buffer,
               idx_buffer, inv_scale, S_q_eff, S_kv_eff);
}
#endif

// The first n values of a key block's object, laid in it flat (value f at
// row f / KV_W, column f % KV_W) and tiled as keys, out into q flat.
static constexpr int32_t KV_W = MHA_KV_HEADS * 2 * DIM_K;
void keep_q(const bfloat16 *obj, bfloat16 *q, int32_t n) {
  event0();
  for (int32_t f = 0; f < n; f += 8) {
    const int32_t r = f / KV_W, c = f % KV_W;
    aie::store_v(q + f, aie::load_v<8>(obj + (r / KV_KEYS) * KV_KEYS * KV_W +
                                       (c / 8) * KV_TILE + (r % KV_KEYS) * 8));
  }
  event1();
}

// KV head kv_head's packed 8-row query block out of q, tiled as attend_block
// reads Q. Rows i and i + 4 take neighbouring heads of the group (heads 2j
// and 2j + 1 of an even group, j and j + 1 of an odd one, its one head twice
// alone), the order one DMA pattern returns O's rows to their heads in.
// Eight rows hold a group of 1 to 6 or 8 so; at 7 or past 8 heads are lost.
void take_q(const bfloat16 *q, bfloat16 *Q, int32_t kv_head, int32_t group) {
  event0();
  const int32_t pairs = group == 1 ? 1 : group % 2 ? group - 1 : group / 2;
  const int32_t step = group == 1 ? 0 : group % 2 ? 1 : 2;
  const bfloat16 *__restrict src = q + kv_head * group * DIM_K;
  for (int32_t i = 0; i < 8; i++) {
    const int32_t head = step * ((i % 4) % pairs) + (i >= 4 && group > 1);
    const bfloat16 *__restrict row = src + head * DIM_K;
    bfloat16 *__restrict dst = Q + (i / 4) * 4 * DIM_K + (i % 4) * 8;
    for (int32_t x = 0; x < DIM_K; x += 8)
      aie::store_v(dst + x * 4, aie::load_v<8>(row + x));
  }
  event1();
}
#endif

// A partial: a query block's float O, unnormalized, then its state, as a
// core of split decode hands its keys' attention on, one object.
static constexpr int32_t PARTIAL = DIM_M * DIM_K + 4 * DIM_M;

// O_b and state_b, another core's attention of the same rows over other
// keys, added into O_a and state_a, each rescaled to the larger running
// maximum by exp2 of the difference, as advance_state rescales a key block.
// A state is as its last key block leaves it, m from 0 and l from 2 * B_q.
static inline void merge_partial(float *O_a, float *state_a, const float *O_b,
                                 const float *state_b, bfloat16 inv_scale) {
  constexpr int32_t B_q = DIM_M, W = STATE_W;
  using Acc = aie::accum<accfloat, W>;
  const auto s = aie::broadcast<bfloat16, W>(inv_scale);
  alignas(64) float f_a[DIM_M], f_b[DIM_M];
  for (int32_t i = 0; i < B_q; i += W) {
#ifdef MHA_F32_SCORES
    // Float maxima, as advance_state_w takes them.
    const auto m =
        aie::max(aie::load_v<W>(state_a + i), aie::load_v<W>(state_b + i));
    const auto factor = [&](const float *own) {
      Acc d;
      d.from_vector(aie::load_v<W>(own + i));
      const Limbs<W> dl = bf16_limbs(aie::sub(d, m));
      Acc diff = aie::mul(dl.hi, s);
      diff = aie::mac(diff, dl.mid, s);
      diff = aie::mac(diff, dl.lo, s);
      return exp2_bf16(diff.template to_vector<float>());
    };
    const auto fa = factor(state_a), fb = factor(state_b);
#else
    Acc m_a, m_b;
    m_a.from_vector(aie::load_v<W>(state_a + i));
    m_b.from_vector(aie::load_v<W>(state_b + i));
    const auto m = aie::max(m_a.template to_vector<bfloat16>(),
                            m_b.template to_vector<bfloat16>());
    const auto factor = [&](const Acc &own) {
      Acc diff =
          aie::msc(aie::mul(own.template to_vector<bfloat16>(), s), m, s);
      return exp2_bf16(diff.template to_vector<float>());
    };
    const auto fa = factor(m_a), fb = factor(m_b);
#endif
    Acc la, lb;
    la.from_vector(aie::load_v<W>(state_a + 2 * B_q + i));
    lb.from_vector(aie::load_v<W>(state_b + 2 * B_q + i));
    const Limbs<W> al = bf16_limbs(la), bl = bf16_limbs(lb);
    Acc l = aie::mul(al.hi, fa);
    l = aie::mac(l, al.mid, fa);
    l = aie::mac(l, al.lo, fa);
    l = aie::mac(l, bl.hi, fb);
    l = aie::mac(l, bl.mid, fb);
    l = aie::mac(l, bl.lo, fb);
    aie::store_v(state_a + i, Acc(m).template to_vector<float>());
    aie::store_v(state_a + 2 * B_q + i, l.template to_vector<float>());
    aie::store_v(f_a + i, Acc(fa).template to_vector<float>());
    aie::store_v(f_b + i, Acc(fb).template to_vector<float>());
  }
  using Acc64 = aie::accum<accfloat, VECTOR_LENGTH>;
  for (int32_t band = 0; band < B_q; band += PV_R) {
    const auto sa = broadcast_by_row(f_a + band).template to_vector<bfloat16>();
    const auto sb = broadcast_by_row(f_b + band).template to_vector<bfloat16>();
    float *__restrict a = O_a + band * DIM_K;
    const float *__restrict b = O_b + band * DIM_K;
    for (int32_t j = 0; j < BAND_VECTORS; j++) {
      Acc64 oa, ob;
      oa.from_vector(aie::load_v<VECTOR_LENGTH>(a));
      ob.from_vector(aie::load_v<VECTOR_LENGTH>(b));
      const Limbs<VECTOR_LENGTH> al = bf16_limbs(oa), bl = bf16_limbs(ob);
      Acc64 o = aie::mul(al.hi, sa);
      o = aie::mac(o, al.mid, sa);
      o = aie::mac(o, al.lo, sa);
      o = aie::mac(o, bl.hi, sb);
      o = aie::mac(o, bl.mid, sb);
      o = aie::mac(o, bl.lo, sb);
      aie::store_v(a, o.template to_vector<float>());
      a += VECTOR_LENGTH;
      b += VECTOR_LENGTH;
    }
  }
}

// Split decode's merge: n partials, one after another, folded in order into
// O and state; rescale_O ends the merged rows. O and state start as
// open_partial leaves them, which the first partial's merge only copies.
void merge_partials(float *O, float *state, const float *partials, int32_t n,
                    bfloat16 inv_scale) {
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);
  for (int32_t k = 0; k < n; k++) {
    const float *O_b = partials + k * PARTIAL;
    merge_partial(O, state, O_b, O_b + DIM_M * DIM_K, inv_scale);
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

// A partial's start where its first key block is not block 0: O zero, the
// state as init_scale_buffer leaves it.
void open_partial(float *O, float *state) {
  event0();
  zero_vectorized<float, DIM_M, DIM_K, false>(O);
  init_state_w<STATE_W>(state, DIM_M);
  event1();
}

// The state into the partial whose O the key blocks accumulated.
void close_partial(const float *state, float *partial) {
  event0();
  float *__restrict out = partial + DIM_M * DIM_K;
  for (int32_t i = 0; i < 4 * DIM_M; i += 16)
    aie::store_v(out + i, aie::load_v<16>(state + i));
  event1();
}

void init_scale_buffer(float *scale_buffer, int32_t size) {
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  init_state_w<STATE_W>(scale_buffer, size);
  ::aie::set_rounding(saved_rounding);
  event1();
}
}

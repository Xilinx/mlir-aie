//===- mha.cc ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../activation/softmax_aie2p.h"

// mha.cc is a single compilation unit that includes the AIE2P mm and softmax
// kernels on every arch (there is no separate link step).  The col-major B
// variants are compiled by passing -DB_COL_MAJ to the compiler; this flag is
// set in the PeanoCompilationRule configuration for this file.
// mm_aie2p.h provides matmul_bf16_bf16 and matmul_scalar_bf16_bf16.
#include "../aie_kernel_utils.h"
#include "../common/zero.h"
#include "mm_aie2p.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define VECTOR_LENGTH 64

#define ROUNDING_MODE aie::rounding_mode::conv_even

// Row-major variants needed by matmul_PV.  Because there is no separate link
// step, all kernel symbols must be defined in this single translation unit.
extern "C" {

// Expanded along n only: mm_aie2p.h's 2x2 expansion keeps four 64-float
// accumulators plus two A and two B tiles live, which does not fit the
// registers. The mac order per output tile, and so the result, is the same.
void matmul_bf16_bf16_rowmaj(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  event0();
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned r = 8, s = 8, t = 8;
  constexpr unsigned rowA = DIM_M / r, colA = DIM_K / s, colB = DIM_N / t;
  static_assert(DIM_M % r == 0);
  static_assert(DIM_K % s == 0);
  static_assert(DIM_N % (2 * t) == 0);
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;

  for (unsigned z = 0; z < rowA; z++) {
    bfloat16 *__restrict pC1 = c_out + (z * colB) * MMUL::size_C;
    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA1 = a_in + (z * colA) * MMUL::size_A;
      const bfloat16 *__restrict pB1 = b_in + j * MMUL::size_B;
      const bfloat16 *__restrict pB2 = b_in + (j + 1) * MMUL::size_B;

      // Load partial results from C for accumulation in-place; zero.cc does
      // the zeroing when a new accumulation starts.
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

      aie::store_v(pC1, C00.to_vector<bfloat16>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.to_vector<bfloat16>());
      pC1 += MMUL::size_C;
    }
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

} // extern "C" (row-major wrappers)

// O is the r=t=8 blocked GEMM output: element (row, col) of the tile sits at
// (row / 8) * 512 + (col / 8) * 64 + (row % 8) * 8 + col % 8. Transposing an
// 8x8 replica of the eight scale values of a block row builds its 64-lane
// scale pattern in one shuffle. Out of line so its two callers share one copy
// of the unrolled body.
static __attribute__((noinline)) void
scale_blocked_rows(bfloat16 *O, const bfloat16 *scale) {
  using Vec8bf16 = aie::vector<bfloat16, 8>;
  using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
  for (int32_t l = 0; l < 8; l++) {
    Vec8bf16 scale_row = aie::load_v<8>(scale + l * 8);
    Vec64bf16 scale_vec =
        aie::transpose(scale_row.grow_replicate<VECTOR_LENGTH>(), 8, 8);
    bfloat16 *row = O + l * 512;
    AIE_LOOP_UNROLL_FULL
    for (int32_t j = 0; j < 8; j++) {
      Vec64bf16 o_vec = aie::load_v<VECTOR_LENGTH>(row + j * VECTOR_LENGTH);
      aie::store_v(row + j * VECTOR_LENGTH,
                   aie::mul(o_vec, scale_vec).to_vector<bfloat16>());
    }
  }
}

// partial_softmax_alias_bf16 over every valid row of a block whose rows are
// one vector wide, reduced eight rows at a time: unzipping two partially
// reduced vectors pairs each row's low half with its high half, reduce_max's
// and reduce_add's pairing, so P and scale_buffer match the per-row call bit
// for bit. Masked lanes are set to lowest in place first. The max is taken
// before scaling, which a positive scale leaves unchanged.
static constexpr int32_t SM_ROWS = 8;
static constexpr int32_t SM_LANES = 16;

// Lane numbers, compared against a row's first masked column, build the
// suffix mask in one vector compare.
alignas(64) static const int16_t sm_lane_idx[VECTOR_LENGTH] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

static inline __attribute__((always_inline)) aie::mask<VECTOR_LENGTH>
suffix_mask(aie::vector<int16_t, VECTOR_LENGTH> lane, int32_t i,
            int32_t valid_cols, bool diagonal) {
  int32_t start = valid_cols;
  if (diagonal && i + 1 < start)
    start = i + 1;
  return aie::ge(lane, aie::broadcast<int16_t, VECTOR_LENGTH>(start));
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

#if AIE_TUNED_AIE2
// 2^-y for y = m - a * s, 16 lanes, from x = [a | m] and neg_scale = [-s | 1]:
// AIE2's bf16 product sums a lane of each half.  exp2_bf16.h's method
// otherwise, with the same cubic in the upper halves, but k = round(y) is not
// clamped.  Shifting it into the exponent field saturates instead (the caller
// sets saturation), and maxdiff clamps the difference at +0, so every y above
// 128, masked lanes included, gives +0 whatever 2^-f came to.  Below -127.5,
// where 2^-y overflows, the result is undefined; the caller's y is at least
// -|m| * 2^-8.
static inline __attribute__((always_inline)) aie::vector<bfloat16, SM_LANES>
exp2_neg_bf16(aie::vector<bfloat16, 2 * SM_LANES> x,
              aie::vector<bfloat16, 2 * SM_LANES> neg_scale,
              aie::accum<accfloat, SM_LANES> magic) {
  using Acc = aie::accum<accfloat, SM_LANES>;
  const auto one = aie::broadcast<bfloat16, SM_LANES>(1.0f);
  // The mac rounds after each product, so y + magic is not one mac: that
  // rounds m - k' + magic with k' = round(a * s).
  Acc y(mul_elem_16_2(x, neg_scale));
  const aie::vector<float, SM_LANES> ym =
      aie::add(magic, y.to_vector<float>()).to_vector<float>();
  const aie::vector<int32_t, SM_LANES> k = aie::sub(
      ym.cast_to<int32_t>(), aie::broadcast<int32_t, SM_LANES>(0x4b400000));
  Acc f(mac_elem_16_2(x, neg_scale, aie::sub(magic, ym)));
  const auto fv = aie::concat(f.to_vector<bfloat16>(), one);
  Acc t(mul_elem_16_2(
      fv, aie::concat(aie::broadcast<bfloat16, SM_LANES>(-0.0555041087f),
                      aie::broadcast<bfloat16, SM_LANES>(0.2402265069f))));
  t = mul_elem_16_2(
      fv, aie::concat(t.to_vector<bfloat16>(),
                      aie::broadcast<bfloat16, SM_LANES>(-0.6931471805f)));
  t = mul_elem_16_2(fv, aie::concat(t.to_vector<bfloat16>(), one));
  Acc r;
  r.from_vector(
      aie::maxdiff(t.to_vector<float>().cast_to<int32_t>(), aie::upshift(k, 23))
          .cast_to<float>());
  return r.to_vector<bfloat16>();
}
#endif

static void partial_softmax_rows(bfloat16 *__restrict A, bfloat16 *__restrict P,
                                 const bfloat16 *__restrict m_prev,
                                 bfloat16 *__restrict m_new,
                                 bfloat16 *__restrict l_new, int32_t rows,
                                 int32_t valid_cols, bool diagonal,
                                 bfloat16 scale) {
  using Vec16bf16 = aie::vector<bfloat16, SM_LANES>;
  using Vec16f = aie::vector<float, SM_LANES>;
  const auto scale_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(scale);
  const auto lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(
      std::numeric_limits<bfloat16>::lowest());
  const auto lane = aie::load_v<VECTOR_LENGTH>(sm_lane_idx);
  auto fold_max = [](auto a, auto b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::max(lo, hi);
  };
  auto fold_add = [](Vec16f a, Vec16f b, unsigned step) {
    auto [lo, hi] = aie::interleave_unzip(a, b, step);
    return aie::add(aie::accum<accfloat, SM_LANES>(lo), hi).to_vector<float>();
  };

  // The row passes run on to a whole group: the extra rows are padding that
  // was filled with lowest, and the caller zeroes their P rows after.
  const int32_t group_rows = (rows + SM_ROWS - 1) & ~(SM_ROWS - 1);

  // Lanes a row discards hold lowest from here on.
  if (diagonal || valid_cols < VECTOR_LENGTH) {
    bfloat16 *__restrict row = A;
    AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
    AIE_LOOP_UNROLL(4)
    for (int32_t i = 0; i < group_rows; i++) {
      aie::store_v(row,
                   aie::select(aie::load_v<VECTOR_LENGTH>(row), lowest_vec,
                               suffix_mask(lane, i, valid_cols, diagonal)));
      row += VECTOR_LENGTH;
    }
  }

  // Rows past the last valid one keep their scale_buffer entries; their P rows
  // are zeroed after. The group loops run over at least two groups, which the
  // caller's block always has, so that the pipeliner may overlap them.
  const int32_t group_end = group_rows > SM_ROWS ? group_rows : 2 * SM_ROWS;
  auto live = [rows](int32_t g) {
    int32_t n = rows - g >= SM_ROWS ? SM_ROWS : rows - g > 0 ? rows - g : 0;
    return aie::mask<SM_ROWS>::from_uint32((1u << n) - 1);
  };
  // Rounding is monotonic, so for a positive scale the maximum of a scaled
  // row is the scaled maximum of the row, and one vector per group is scaled.
  const bfloat16 *__restrict a = A;
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int32_t g = 0; g < group_end; g += SM_ROWS) {
    Vec16bf16 o = fold_rows<bfloat16, SM_VEC_LEN>(
                      [&](int32_t r) {
                        auto row =
                            aie::load_v<VECTOR_LENGTH>(a + r * VECTOR_LENGTH);
                        return aie::max(row.extract<SM_VEC_LEN>(0),
                                        row.extract<SM_VEC_LEN>(1));
                      },
                      fold_max)
                      .extract<SM_LANES>(0);
    a += SM_ROWS * VECTOR_LENGTH;
    auto scaled = aie::mul(o, aie::broadcast<bfloat16, SM_LANES>(scale))
                      .to_vector<bfloat16>()
                      .extract<SM_ROWS>(0);
    auto m = aie::max(aie::max(lowest_vec.extract<SM_ROWS>(0), scaled),
                      aie::load_v<SM_ROWS>(m_prev + g));
    aie::store_v(m_new + g,
                 aie::select(aie::load_v<SM_ROWS>(m_new + g), m, live(g)));
  }

  // Each row's exponentials go to P.  A discarded lane's is +0.
  bfloat16 *__restrict p = P;
  const bfloat16 *__restrict m = m_new;
#if AIE_TUNED_AIE2
  // Half a row per pass keeps the loop within the registers; a whole row
  // spills.
  const auto neg_scale = aie::concat(aie::broadcast<bfloat16, SM_LANES>(-scale),
                                     aie::broadcast<bfloat16, SM_LANES>(1.0f));
  aie::accum<accfloat, SM_LANES> magic;
  magic.from_vector(aie::broadcast<float, SM_LANES>(12582912.0f));
  aie::saturation_mode saved_saturation =
      aie::swap_saturation(aie::saturation_mode::saturate);
  for (int32_t h = 0; h < VECTOR_LENGTH; h += 2 * SM_LANES) {
    a = A + h;
    p = P + h;
    m = m_new;
    AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
    for (int32_t i = 0; i < group_rows; i++) {
      const auto m_vec = aie::broadcast<bfloat16, SM_LANES>(*m++);
      aie::store_v(p,
                   exp2_neg_bf16(aie::concat(aie::load_v<SM_LANES>(a), m_vec),
                                 neg_scale, magic));
      aie::store_v(
          p + SM_LANES,
          exp2_neg_bf16(aie::concat(aie::load_v<SM_LANES>(a + SM_LANES), m_vec),
                        neg_scale, magic));
      a += VECTOR_LENGTH;
      p += VECTOR_LENGTH;
    }
  }
  aie::set_saturation(saved_saturation);
#else
  // Taking m off as m * 1 in the multiplier gives the same difference as
  // subtracting it, without widening m to 64 accumulator lanes first.
  const auto one_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(1.0f);
  a = A;
  AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
  AIE_LOOP_UNROLL(2)
  for (int32_t i = 0; i < group_rows; i++) {
    aie::accum<accfloat, VECTOR_LENGTH> exp_in =
        aie::msc(aie::mul(aie::load_v<VECTOR_LENGTH>(a), scale_vec),
                 aie::broadcast<bfloat16, VECTOR_LENGTH>(*m++), one_vec);
    aie::store_v(p, exp2_bf16(exp_in.to_vector<float>()));
    a += VECTOR_LENGTH;
    p += VECTOR_LENGTH;
  }
#endif

  // Their sums, narrowed to 16 lanes, are parked in the A rows, which nothing
  // reads after this call.  Doing this in the loop above would make each
  // row's load wait on the previous row's store.
  p = P;
  bfloat16 *__restrict sums = A;
  AIE_LOOP_MIN_ITERATION_COUNT(SM_ROWS)
  AIE_LOOP_UNROLL(2)
  for (int32_t i = 0; i < group_rows; i++) {
    auto exp_val = aie::load_v<VECTOR_LENGTH>(p);
    p += VECTOR_LENGTH;
    // No lane is -0, so starting from the first half is starting from zero.
    aie::accum<accfloat, SM_VEC_LEN> sum(exp_val.extract<SM_VEC_LEN>(0));
    auto s = aie::add(sum, exp_val.extract<SM_VEC_LEN>(1)).to_vector<float>();
    Vec16f half =
        aie::add(aie::accum<accfloat, SM_LANES>(s.extract<SM_LANES>(0)),
                 s.extract<SM_LANES>(1))
            .to_vector<float>();
    aie::store_v(sums, aie::vector_cast<bfloat16>(half));
    sums += VECTOR_LENGTH;
  }

  sums = A;
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int32_t g = 0; g < group_end; g += SM_ROWS) {
    Vec16f t = fold_rows<float, SM_LANES>(
        [&](int32_t r) {
          return aie::vector_cast<float>(
              aie::load_v<2 * SM_LANES>(sums + r * VECTOR_LENGTH));
        },
        fold_add);
    sums += SM_ROWS * VECTOR_LENGTH;
    aie::accum<accfloat, SM_ROWS> l(t.extract<SM_ROWS>(0));
    aie::store_v(l_new + g, aie::select(aie::load_v<SM_ROWS>(l_new + g),
                                        l.to_vector<bfloat16>(), live(g)));
  }
}

extern "C" {
void partial_softmax_bf16(bfloat16 *input, bfloat16 *output,
                          bfloat16 *scale_buffer, const int32_t input_size,
                          const int32_t row_idx, const int32_t row_size,
                          const bfloat16 scale);
void passThroughLine(int32_t *in, int32_t *out, int32_t lineWidth);

void matmul_bf16_bf16_wrapper(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out,
                              int32_t *idx_buffer) {
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  if (idx_buffer[0] > idx_buffer[1]) {
    // matmul_bf16_bf16 brackets itself; an empty pair keeps a masked call to
    // one interval too, so every call can be timed.
    event0();
    event1();
    ::aie::set_rounding(saved_rounding);
    return;
  }

  matmul_bf16_bf16(a_in, b_in, c_out);
  ::aie::set_rounding(saved_rounding);
}

void matmul_bf16_bf16_wrapper_scalar(bfloat16 *a_in, bfloat16 *b_in,
                                     bfloat16 *c_out) {
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);
  matmul_scalar_bf16_bf16(a_in, b_in, c_out);
  ::aie::set_rounding(saved_rounding);
}

void matmul_PV(bfloat16 *Q, bfloat16 *K, bfloat16 *out, bfloat16 *scale_buffer,
               const int32_t B_q, int32_t first_iter, int32_t *idx_buffer) {
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  if (idx_buffer[0] > idx_buffer[1]) {
    ::aie::set_rounding(saved_rounding);
    return;
  }

  // 64 emul: O dims = [(8, 512), (8, 8), (8, 64), (8, 1)]
  // VJUNG: Scale O_{i-1} by 1/exp(m_{i-1} - m_{i}) store in
  // scale_buffer[3*B_q:3*B_q + B_q] VJUNG: Skip this for the first iteration as
  // 1/exp(m_{i-1} - m_{i}) degenerates to inf due to m intizalized to -inf
  if (first_iter != 0) {
    scale_blocked_rows(out, scale_buffer + 3 * B_q);
  }

  matmul_bf16_bf16_rowmaj(Q, K, out);
  ::aie::set_rounding(saved_rounding);
}

void rescale_O(bfloat16 *O, bfloat16 *scale_buffer, int32_t B_q,
               int32_t *idx_buffer) {
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  for (int32_t i = 0; i < B_q; i += VECTOR_LENGTH) {
    using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
    Vec64bf16 l_vec = aie::load_v<VECTOR_LENGTH>(scale_buffer + 2 * B_q + i);
    l_vec = aie::inv(l_vec);
    aie::store_v(scale_buffer + 2 * B_q + i, l_vec);
  }

  // VJUNG: Only after all KV are processed
  // VJUNG: TODO: Make this generic for every tile size
  // VJUNG: Need to scale depending on the data layout at the output of GEMM
  // VJUNG: Scale O_{i} by 1/l_{i}
  scale_blocked_rows(O, scale_buffer + 2 * B_q);
  ::aie::set_rounding(saved_rounding);
  event1();
}

void partial_softmax(bfloat16 *A, bfloat16 *P, bfloat16 *scale_buffer,
                     int32_t *idx_buffer, bfloat16 inv_scale, int32_t B_q,
                     int32_t B_kv, int32_t S_q_eff, int32_t S_kv_eff) {
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  // Block indices
  int32_t q_block_idx = idx_buffer[1];
  int32_t kv_block_idx = idx_buffer[0];

  // Causal full mask: skip blocks strictly above diagonal
  if (kv_block_idx > q_block_idx) {
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

  using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
  Vec64bf16 lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(
      std::numeric_limits<bfloat16>::lowest());

  // Tail mask: invalidate padded Q rows.  They are the end of A, so the walk
  // over (row, vector) is one linear fill.
  for (int32_t n = valid_q_rows * B_kv; n < B_q * B_kv; n += VECTOR_LENGTH) {
    aie::store_v(A + n, lowest_vec);
  }

  // partial_softmax_rows scales a row's maximum instead of every element,
  // which gives the same maximum only for a positive, finite scale.
  uint16_t scale_bits = __builtin_bit_cast(uint16_t, inv_scale);
  if (B_kv == VECTOR_LENGTH && B_q % SM_ROWS == 0 && B_q >= 2 * SM_ROWS &&
      scale_bits > 0 && scale_bits < 0x7f80) {
    partial_softmax_rows(A, P, scale_buffer, scale_buffer + B_q,
                         scale_buffer + 3 * B_q, valid_q_rows, valid_kv_cols,
                         kv_block_idx == q_block_idx, inv_scale);
  } else {
    // Everything a valid row discards is a suffix of that row: the columns from
    // valid_kv_cols on, and on the diagonal block the columns past the
    // diagonal, so the earlier start covers both. A lane mask covers a suffix
    // that starts mid-vector; its bits are built directly because mask's own
    // runtime shift is a loop.
    if (valid_kv_cols < B_kv || kv_block_idx == q_block_idx) {
      for (int32_t i = 0; i < valid_q_rows; i++) {
        int32_t start = valid_kv_cols;
        if (kv_block_idx == q_block_idx && i + 1 < start) {
          start = i + 1;
        }
        int32_t base = start & ~(VECTOR_LENGTH - 1);
        if (base < B_kv) {
          bfloat16 *row = A + i * B_kv;
          aie::mask<VECTOR_LENGTH> above =
              aie::mask<VECTOR_LENGTH>::from_uint64(~uint64_t(0)
                                                    << (start - base));
          aie::store_v(row + base,
                       aie::select(aie::load_v<VECTOR_LENGTH>(row + base),
                                   lowest_vec, above));
          for (int32_t c = base + VECTOR_LENGTH; c < B_kv; c += VECTOR_LENGTH) {
            aie::store_v(row + c, lowest_vec);
          }
        }
      }
    }

    // The alias form, not the partial_softmax_bf16 entry point, which brackets
    // each row with markers of its own.
    int32_t i = 0;
    for (; i + 4 <= valid_q_rows; i += 4) {
      partial_softmax_alias_bf16(A + B_kv * i, P + B_kv * i, scale_buffer, B_kv,
                                 i, B_q, inv_scale);
      partial_softmax_alias_bf16(A + B_kv * (i + 1), P + B_kv * (i + 1),
                                 scale_buffer, B_kv, i + 1, B_q, inv_scale);
      partial_softmax_alias_bf16(A + B_kv * (i + 2), P + B_kv * (i + 2),
                                 scale_buffer, B_kv, i + 2, B_q, inv_scale);
      partial_softmax_alias_bf16(A + B_kv * (i + 3), P + B_kv * (i + 3),
                                 scale_buffer, B_kv, i + 3, B_q, inv_scale);
    }
    for (; i < valid_q_rows; i++) {
      partial_softmax_alias_bf16(A + B_kv * i, P + B_kv * i, scale_buffer, B_kv,
                                 i, B_q, inv_scale);
    }
  }
  // Zero out P rows corresponding to padded Q rows, which are again a suffix
  // and so one linear fill.
  Vec64bf16 zeros_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(0.0f);
  for (int32_t n = valid_q_rows * B_kv; n < B_q * B_kv; n += VECTOR_LENGTH) {
    aie::store_v(P + n, zeros_vec);
  }

  for (int32_t i = 0; i < B_q; i += VECTOR_LENGTH) {

    Vec64bf16 m_i_minus_1 = aie::load_v<VECTOR_LENGTH>(scale_buffer + i);
    Vec64bf16 m_i = aie::load_v<VECTOR_LENGTH>(scale_buffer + B_q + i);
    Vec64bf16 l_i_minus_1 =
        aie::load_v<VECTOR_LENGTH>(scale_buffer + 2 * B_q + i);
    Vec64bf16 accum_exp_val =
        aie::load_v<VECTOR_LENGTH>(scale_buffer + 3 * B_q + i);

    aie::accum<accfloat, VECTOR_LENGTH> l_i_accum =
        aie::zeros<accfloat, VECTOR_LENGTH>();

    aie::accum<accfloat, VECTOR_LENGTH> diff =
        aie::accum<accfloat, VECTOR_LENGTH>(aie::sub(m_i_minus_1, m_i));
    l_i_accum.from_vector(exp2_bf16(diff.to_vector<float>()));
    Vec64bf16 max_diff_exp = l_i_accum.to_vector<bfloat16>();

    aie::store_v(scale_buffer + 3 * B_q + i, max_diff_exp);
    aie::accum<accfloat, VECTOR_LENGTH> l_i =
        aie::add(aie::mul(max_diff_exp, l_i_minus_1), accum_exp_val);
    aie::store_v(scale_buffer + 2 * B_q + i, l_i.to_vector<bfloat16>());
    aie::store_v(scale_buffer + i, m_i);
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

void init_scale_buffer(bfloat16 *scale_buffer, int32_t size) {
  event0();
  ::aie::rounding_mode saved_rounding = ::aie::swap_rounding(ROUNDING_MODE);

  using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
  Vec64bf16 lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(
      std::numeric_limits<bfloat16>::lowest());
  Vec64bf16 zeros_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(0.0f);

  for (int32_t i = 0; i < size; i += VECTOR_LENGTH) {
    // VJUNG: m_{i-1} vector
    aie::store_v(scale_buffer + i, lowest_vec);
    // VJUNG: m_{i} vector
    aie::store_v(scale_buffer + size + i, zeros_vec);
    // VJUNG: l_{i} vector
    aie::store_v(scale_buffer + 2 * size + i, zeros_vec);
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}
}

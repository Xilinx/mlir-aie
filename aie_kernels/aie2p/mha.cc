//===- mha.cc ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "softmax.cc"

// mha.cc is a single compilation unit that includes mm.cc and softmax.cc via
// #include (there is no separate link step).  The col-major B variants are
// compiled by passing -DB_COL_MAJ to the compiler; this flag is set in the
// PeanoCompilationRule configuration for this file.
// mm.cc provides matmul_bf16_bf16 and matmul_scalar_bf16_bf16.
#include "../aie_kernel_utils.h"
#include "../generic/zero.cc"
#include "mm.cc"

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

// The 8x8x8 bf16 mmul keeps a 64-float accumulator, so mm.cc's 2x2 expansion
// holds four of them -- eight accumulator registers -- plus two A and two B
// tiles across the reduction.  That does not fit, and the reduction loop
// schedules at II 175 for its eight native macs.  Expanding only along n keeps
// two accumulators live and one A tile, and the same eight macs schedule at
// II 36, which is the rate mm.cc's own bf16 product runs at.  The mac order
// per output tile is unchanged, so the result is unchanged too.
void matmul_bf16_bf16_rowmaj(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
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
}

} // extern "C" (row-major wrappers)

// O is the r=t=8 blocked GEMM output: element (row, col) of the tile sits at
// (row / 8) * 512 + (col / 8) * 64 + (row % 8) * 8 + col % 8, so one row's
// scale covers only eight lanes at a time and the tile costs 512 broadcasts
// and 512 eight-lane multiplies.  Transposing an 8x8 replica of the eight
// scale values of a block row builds the whole 64-lane pattern in one shuffle,
// which turns the block row into eight contiguous 64-lane multiplies.  Kept
// out of line so its two callers share one copy of the unrolled body.
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

  using Vec64bf16 = aie::vector<bfloat16, VECTOR_LENGTH>;
  Vec64bf16 lowest_vec = aie::broadcast<bfloat16, VECTOR_LENGTH>(
      std::numeric_limits<bfloat16>::lowest());

  // Tail mask: invalidate padded Q rows.  They are the end of A, so the walk
  // over (row, vector) is one linear fill.
  for (int32_t n = valid_q_rows * B_kv; n < B_q * B_kv; n += VECTOR_LENGTH) {
    aie::store_v(A + n, lowest_vec);
  }

  // Everything a valid row discards is a suffix of that row: the columns from
  // valid_kv_cols on, and on the diagonal block the columns past the diagonal.
  // The two start at different columns but both run to the end of the row, so
  // the earlier start covers both.  A suffix starting mid-vector used to fall
  // entirely to the scalar remainder loop -- on a 64-wide diagonal block that
  // is sum(B_kv - 1 - i) single-element stores -- and a lane mask keeps it on
  // the vector path.  mask's own runtime shift walks its backing words in a
  // loop the target keeps as a loop, so build the bits directly.
  if (valid_kv_cols < B_kv || kv_block_idx == q_block_idx) {
    for (int32_t i = 0; i < valid_q_rows; i++) {
      int32_t start = valid_kv_cols;
      if (kv_block_idx == q_block_idx && i + 1 < start) {
        start = i + 1;
      }
      int32_t base = start & ~(VECTOR_LENGTH - 1);
      if (base < B_kv) {
        bfloat16 *row = A + i * B_kv;
        aie::mask<VECTOR_LENGTH> above = aie::mask<VECTOR_LENGTH>::from_uint64(
            ~uint64_t(0) << (start - base));
        aie::store_v(row + base,
                     aie::select(aie::load_v<VECTOR_LENGTH>(row + base),
                                 lowest_vec, above));
        for (int32_t c = base + VECTOR_LENGTH; c < B_kv; c += VECTOR_LENGTH) {
          aie::store_v(row + c, lowest_vec);
        }
      }
    }
  }

  int32_t i = 0;
  for (; i + 4 <= valid_q_rows; i += 4) {
    partial_softmax_bf16(A + B_kv * i, P + B_kv * i, scale_buffer, B_kv, i, B_q,
                         inv_scale);
    partial_softmax_bf16(A + B_kv * (i + 1), P + B_kv * (i + 1), scale_buffer,
                         B_kv, i + 1, B_q, inv_scale);
    partial_softmax_bf16(A + B_kv * (i + 2), P + B_kv * (i + 2), scale_buffer,
                         B_kv, i + 2, B_q, inv_scale);
    partial_softmax_bf16(A + B_kv * (i + 3), P + B_kv * (i + 3), scale_buffer,
                         B_kv, i + 3, B_q, inv_scale);
  }
  for (; i < valid_q_rows; i++) {
    partial_softmax_bf16(A + B_kv * i, P + B_kv * i, scale_buffer, B_kv, i, B_q,
                         inv_scale);
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
    l_i_accum = aie::exp2<bfloat16>(diff.to_vector<float>());
    Vec64bf16 max_diff_exp = l_i_accum.to_vector<bfloat16>();

    aie::store_v(scale_buffer + 3 * B_q + i, max_diff_exp);
    aie::accum<accfloat, VECTOR_LENGTH> l_i =
        aie::add(aie::mul(max_diff_exp, l_i_minus_1), accum_exp_val);
    aie::store_v(scale_buffer + 2 * B_q + i, l_i.to_vector<bfloat16>());
    aie::store_v(scale_buffer + i, m_i);
  }
  ::aie::set_rounding(saved_rounding);
}

void init_scale_buffer(bfloat16 *scale_buffer, int32_t size) {
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
}
}

//===- attention_i8.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// One head of int8 attention over a 2-D map (YOLO26n's PSA), a row of
// queries at a time; see attention_i8 in python/iron/kernels/linalg.py.
// Rows are [C/8][W][8]. attention_i8_load_kv stores a row's keys as
// [N/8][KD][8] (8 keys of every dimension per mmul B tile) and its values as
// [HD/8][N][8].
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef ATTN_SEQ
#error                                                                         \
    "ATTN_SEQ (keys), ATTN_KEY_DIM, ATTN_HEAD_DIM and ATTN_WIDTH must be defined"
#endif

constexpr int N = ATTN_SEQ;
constexpr int KD = ATTN_KEY_DIM;
constexpr int HD = ATTN_HEAD_DIM;
constexpr int W = ATTN_WIDTH;
static_assert(N % 8 == 0 && KD % 8 == 0 && HD % 8 == 0 && W % 4 == 0,
              "attention_i8 tiles keys and dimensions by 8, queries by 4");

using MMUL = aie::mmul<4, 8, 8, int8, int8>;

extern "C" {

void attention_i8_load_kv(const int8_t *__restrict k_row,
                          const int8_t *__restrict v_row, int8_t *__restrict kt,
                          int8_t *__restrict v, int32_t row) {
  event0();
  for (int d = 0; d < KD; d++)
    for (int x = 0; x < W; x++) {
      const int j = row * W + x;
      kt[(j / 8) * KD * 8 + d * 8 + j % 8] =
          k_row[(d / 8) * W * 8 + x * 8 + d % 8];
    }
  for (int g = 0; g < HD / 8; g++)
    for (int i = 0; i < W * 8; i += 32)
      aie::store_v(v + g * N * 8 + row * W * 8 + i,
                   aie::load_v<32>(v_row + g * W * 8 + i));
  event1();
}

void attention_i8(const int8_t *__restrict q_row, const int8_t *__restrict kt,
                  const int8_t *__restrict v,
                  const int32_t *__restrict exp_table,
                  int8_t *__restrict out_row, int32_t q_mul, int32_t q_shift,
                  int32_t score_shift, int32_t prob_shift, int32_t out_shift) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  alignas(64) int8_t q[KD * W];
  // Scores and probabilities by key block: [N/8][4 queries][8 keys], the
  // layout an mmul A operand reads 8 keys of 4 queries from.
  alignas(64) int8_t s[N * 4];
  alignas(64) int8_t p[N * 4];
  // Each score's gap below its query's top, 0 to 255: it wraps exactly into
  // a byte.
  alignas(64) uint8_t gaps[N * 4];

  const aie::vector<int8, 32> mul = aie::broadcast<int8, 32>(q_mul);
  for (int i = 0; i < KD * W; i += 32)
    aie::store_v(q + i, aie::mul(aie::load_v<32>(q_row + i), mul)
                            .template to_vector<int8>(q_shift));

  for (int qb = 0; qb < W / 4; qb++) {
    for (int kb = 0; kb < N / 8; kb++) {
      MMUL acc;
      acc = aie::zeros<acc32, 32>();
      for (int g = 0; g < KD / 8; g++)
        acc.mac(aie::load_v<32>(q + g * W * 8 + qb * 32),
                aie::load_v<64>(kt + kb * KD * 8 + g * 64));
      aie::store_v(s + kb * 32, acc.template to_vector<int8>(score_shift));
    }

    aie::vector<int8, 32> highest = aie::load_v<32>(s);
    for (int kb = 1; kb < N / 8; kb++)
      highest = aie::max(highest, aie::load_v<32>(s + kb * 32));
    alignas(32) int8_t tops[32];
    aie::store_v(tops, highest);
    for (int i = 0; i < 4; i++) {
      int8_t top = tops[i * 8];
      for (int c = 1; c < 8; c++)
        top = tops[i * 8 + c] > top ? tops[i * 8 + c] : top;
      for (int c = 0; c < 8; c++)
        tops[i * 8 + c] = top;
    }
    const aie::vector<int8, 32> top_lanes = aie::load_v<32>(tops);
    for (int kb = 0; kb < N / 8; kb++)
      aie::store_v(reinterpret_cast<int8_t *>(gaps) + kb * 32,
                   aie::sub(top_lanes, aie::load_v<32>(s + kb * 32)));
    for (int i = 0; i < 4; i++) {
      // Each entry is below 2**29, so 8 of them sum in 32 bits.
      int64_t total = 0;
      for (int kb = 0; kb < N / 8; kb++) {
        const uint8_t *gap = gaps + kb * 32 + i * 8;
        uint32_t block = 0;
        for (int c = 0; c < 8; c++)
          block += exp_table[gap[c]];
        total += block;
      }
      // A probability's quotient is at most 128, so the top gap's divides by
      // restoring division on total * 2**b: 64-bit division is a libcall
      // Peano lacks.
      int64_t steps[8];
      steps[0] = total;
      for (int b = 1; b < 8; b++)
        steps[b] = steps[b - 1] + steps[b - 1];
      int64_t rem = exp_table[0];
      for (int b = 0; b < prob_shift; b++)
        rem += rem;
      int level = 0;
      for (int b = 7; b >= 0; b--)
        if (rem >= steps[b]) {
          rem -= steps[b];
          level |= 1 << b;
        }
      level += rem + rem > total || (rem + rem == total && (level & 1));
      // A probability depends only on its gap below the top score, and the
      // table falls, so the rounded probability falls with the gap: walk the
      // levels down, giving each gap the highest one it reaches. A gap
      // reaches `level` when exp_table[gap] * 2**k, k = prob_shift + 1,
      // exceeds bar = (2 * level - 1) * total, or equals it with `level` even
      // (half to even); exp_table[gap] * 2**k being a multiple of 2**k, that
      // is exp_table[gap] > (bar >> k) - (bar % 2**k == 0 && level even).
      // bar >> k starts at most exp_table[0] and only falls; below 0, every
      // gap left reaches the level. So the test is 32-bit.
      const int k = prob_shift + 1;
      const int64_t unit = (int64_t)1 << k;
      int64_t bar = 0;
      for (int b = 0; b < 8; b++)
        if ((2 * level - 1) >> b & 1)
          bar += steps[b];
      int64_t bar_high = bar >> k;
      int64_t bar_low = bar & (unit - 1);
      alignas(32) int8_t by_gap[256];
      for (int d = 0; d < 256; d += 32)
        aie::store_v(by_gap + d, aie::zeros<int8, 32>());
      for (int d = 0; level > 0 && d < 256; level--) {
        const int32_t bound =
            bar_high < 0 ? -1 : bar_high - (bar_low == 0 && (level & 1) == 0);
        const int8_t value = level > 127 ? 127 : level;
        while (d < 256 && exp_table[d] > bound)
          by_gap[d++] = value;
        // One level down, bar falls by 2 * total.
        bar_low -= steps[1] & (unit - 1);
        bar_high -= steps[1] >> k;
        if (bar_low < 0) {
          bar_low += unit;
          bar_high--;
        }
      }
      for (int kb = 0; kb < N / 8; kb++)
        for (int c = 0; c < 8; c++)
          p[kb * 32 + i * 8 + c] = by_gap[gaps[kb * 32 + i * 8 + c]];
    }

    for (int g = 0; g < HD / 8; g++) {
      MMUL acc;
      acc = aie::zeros<acc32, 32>();
      for (int kb = 0; kb < N / 8; kb++)
        acc.mac(aie::load_v<32>(p + kb * 32),
                aie::load_v<64>(v + g * N * 8 + kb * 64));
      aie::store_v(out_row + g * W * 8 + qb * 32,
                   acc.template to_vector<int8>(out_shift));
    }
  }
  event1();
}

} // extern "C"

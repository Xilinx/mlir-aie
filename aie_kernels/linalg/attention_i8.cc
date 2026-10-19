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

    for (int i = 0; i < 4; i++) {
      int top = -128;
      for (int kb = 0; kb < N / 8; kb++)
        for (int c = 0; c < 8; c++)
          top = s[kb * 32 + i * 8 + c] > top ? s[kb * 32 + i * 8 + c] : top;
      int64_t total = 0;
      for (int kb = 0; kb < N / 8; kb++)
        for (int c = 0; c < 8; c++)
          total += exp_table[top - s[kb * 32 + i * 8 + c]];
      // A probability's quotient is at most 128, so it divides by restoring
      // division on total * 2**b: 64-bit division is a libcall Peano lacks.
      int64_t steps[8];
      steps[0] = total;
      for (int b = 1; b < 8; b++)
        steps[b] = steps[b - 1] + steps[b - 1];
      for (int kb = 0; kb < N / 8; kb++)
        for (int c = 0; c < 8; c++) {
          int64_t rem = exp_table[top - s[kb * 32 + i * 8 + c]];
          for (int b = 0; b < prob_shift; b++)
            rem += rem;
          int32_t prob = 0;
          // Below one half rounds to 0 (a tie too, 0 being even).
          if (rem + rem > total) {
            for (int b = 7; b >= 0; b--)
              if (rem >= steps[b]) {
                rem -= steps[b];
                prob |= 1 << b;
              }
            prob += rem + rem > total || (rem + rem == total && (prob & 1));
            prob = prob > 127 ? 127 : prob;
          }
          p[kb * 32 + i * 8 + c] = prob;
        }
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

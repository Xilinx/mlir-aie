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

#if __AIE_ARCH__ >= 21
// Tables as AIE2P's 4-way LUT read takes them: an int16 per entry, each run of
// 16 entries twice over (the read takes a lane's parity for address bit 5),
// where the read can reach them (the stack it cannot). The exp table in its
// low and high 16 bits; each of a block's 4 queries' probabilities by gap.
alignas(64) static int16_t exp_lo_runs[512];
alignas(64) static int16_t exp_hi_runs[512];
alignas(64) static int16_t prob_runs[4 * 512];

// The entries of 32 gaps from a table's runs whose entry 128 is at `base`,
// each lane's offset moved on by `first` (lanes 0 to 15) or `second` (16 to
// 31) bytes: to its query's table, when a block's queries have one each.
static inline __attribute__((always_inline)) aie::vector<int16, 32>
read_runs(const int8_t *base, aie::vector<uint8, 32> gaps,
          aie::vector<int32, 16> first, aie::vector<int32, 16> second) {
  // A gap as int8 is gap - 128: entry gap, 4 bytes an entry of offset.
  aie::accum<acc64, 32> scaled;
  scaled.from_vector(
      aie::bit_xor(gaps.cast_to<int8>(), aie::broadcast<int8, 32>(-128)), 2);
  const aie::vector<int32, 16> lanes0 = aie::add(
      scaled.template extract<16>(0).template to_vector<int32>(0), first);
  const aie::vector<int32, 16> lanes1 = aie::add(
      scaled.template extract<16>(1).template to_vector<int32>(0), second);
  v64int8 r0, r1, r2, r3;
  load_lut_2x_int8(base, base, lanes0, r0, r1);
  load_lut_2x_int8(base, base, lanes1, r2, r3);
  return aie::vector<int16, 32>(
      (v32int16)::shuffle(::shuffle(r0, r1, T16_16x4_lo),
                          ::shuffle(r2, r3, T16_16x4_lo), T256_2x2_lo));
}

#endif

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

#if __AIE_ARCH__ >= 21
  for (int r = 0; r < 16; r++) {
    // An int32 entry is its low then its high 16 bits.
    const aie::vector<int16, 32> halves =
        aie::load_v<16>(exp_table + 16 * r).cast_to<int16>();
    const aie::vector<int16, 16> lo = aie::filter_even(halves);
    const aie::vector<int16, 16> hi = aie::filter_odd(halves);
    aie::store_v(exp_lo_runs + 32 * r, lo);
    aie::store_v(exp_lo_runs + 32 * r + 16, lo);
    aie::store_v(exp_hi_runs + 32 * r, hi);
    aie::store_v(exp_hi_runs + 32 * r + 16, hi);
  }
  const int8_t *exp_lo = reinterpret_cast<const int8_t *>(exp_lo_runs) + 512;
  const int8_t *exp_hi = reinterpret_cast<const int8_t *>(exp_hi_runs) + 512;
  const int8_t *probs = reinterpret_cast<const int8_t *>(prob_runs) + 512;
  // A table's runs take 1024 bytes; queries 1, 2 and 3 of a block read the
  // ones after query 0's.
  alignas(64) static const int32_t query_tables[32] = {
      0,    0,    0,    0,    0,    0,    0,    0,    1024, 1024, 1024,
      1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 2048, 2048,
      2048, 2048, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072};
  const aie::vector<int32, 16> same = aie::zeros<int32, 16>();
  const aie::vector<int32, 16> tables_lo = aie::load_v<16>(query_tables);
  const aie::vector<int32, 16> tables_hi = aie::load_v<16>(query_tables + 16);
#endif

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
#if __AIE_ARCH__ >= 21
    // Each lane sums its keys' exp entries in low and high 16-bit halves:
    // N / 8 halves of at most 2**16 fit a lane's 32 bits.
    aie::vector<uint32, 32> lo_sum = aie::zeros<uint32, 32>();
    aie::vector<uint32, 32> hi_sum = aie::zeros<uint32, 32>();
    for (int kb = 0; kb < N / 8; kb++) {
      const aie::vector<uint8, 32> g = aie::load_v<32>(gaps + kb * 32);
      lo_sum = aie::add(
          lo_sum, read_runs(exp_lo, g, same, same).cast_to<uint16>().unpack());
      hi_sum = aie::add(
          hi_sum, read_runs(exp_hi, g, same, same).cast_to<uint16>().unpack());
    }
    alignas(64) uint32_t lo_lanes[32], hi_lanes[32];
    aie::store_v(lo_lanes, lo_sum);
    aie::store_v(hi_lanes, hi_sum);
#endif
    for (int i = 0; i < 4; i++) {
      int64_t total = 0;
#if __AIE_ARCH__ >= 21
      for (int c = 0; c < 8; c++)
        total += ((int64_t)hi_lanes[i * 8 + c] << 16) + lo_lanes[i * 8 + c];
#else
      // Each entry is below 2**29, so 8 of them sum in 32 bits.
      for (int kb = 0; kb < N / 8; kb++) {
        const uint8_t *gap = gaps + kb * 32 + i * 8;
        uint32_t block = 0;
        for (int c = 0; c < 8; c++)
          block += exp_table[gap[c]];
        total += block;
      }
#endif
      // A probability's quotient is at most 128, so the top gap's divides by
      // restoring division on total * 2**b: 64-bit division is a libcall
      // Peano lacks.
      int64_t steps[8];
      AIE_LOOP_UNROLL_FULL
      for (int b = 0; b < 8; b++)
        steps[b] = total << b;
      int64_t rem = (int64_t)exp_table[0] << prob_shift;
      int level = 0;
      // Selects, not branches.
      AIE_LOOP_UNROLL_FULL
      for (int b = 7; b >= 0; b--) {
        const bool take = rem >= steps[b];
        rem = take ? rem - steps[b] : rem;
        level |= (int)take << b;
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
      AIE_LOOP_UNROLL_FULL
      for (int b = 0; b < 8; b++)
        bar += (2 * level - 1) >> b & 1 ? steps[b] : 0;
      // So bar >> k fits 32 bits until it goes below 0, the last level the
      // walk takes, where only its sign counts: a fall of 2**30 or more takes
      // it there as surely as the whole one.
      const uint32_t mask = (uint32_t)(unit - 1);
      const int64_t fall = steps[1] >> k;
      const int32_t fall_high = fall > (1 << 30) ? (1 << 30) : (int32_t)fall;
      const uint32_t fall_low = (uint32_t)steps[1] & mask;
      int32_t bar_high = (int32_t)(bar >> k);
      uint32_t bar_low = (uint32_t)bar & mask;
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
        bar_high -= fall_high + (bar_low < fall_low);
        bar_low = (bar_low - fall_low) & mask;
      }
#if __AIE_ARCH__ >= 21
      for (int r = 0; r < 16; r++) {
        const aie::vector<int16, 16> run =
            aie::load_v<16>(by_gap + 16 * r).unpack();
        aie::store_v(prob_runs + i * 512 + 32 * r, run);
        aie::store_v(prob_runs + i * 512 + 32 * r + 16, run);
      }
#else
      for (int kb = 0; kb < N / 8; kb++)
        for (int c = 0; c < 8; c++)
          p[kb * 32 + i * 8 + c] = by_gap[gaps[kb * 32 + i * 8 + c]];
#endif
    }
#if __AIE_ARCH__ >= 21
    // The reads cannot be told apart from the stores, so a group's blocks are
    // all read before any is stored.
    constexpr int GROUP = 5;
    for (int kb = 0; kb < N / 8 / GROUP * GROUP; kb += GROUP) {
      aie::vector<int16, 32> read[GROUP];
      AIE_LOOP_UNROLL_FULL
      for (int j = 0; j < GROUP; j++)
        read[j] = read_runs(probs, aie::load_v<32>(gaps + (kb + j) * 32),
                            tables_lo, tables_hi);
      AIE_LOOP_UNROLL_FULL
      for (int j = 0; j < GROUP; j++)
        aie::store_v(p + (kb + j) * 32, read[j].pack());
    }
    for (int kb = N / 8 / GROUP * GROUP; kb < N / 8; kb++)
      aie::store_v(p + kb * 32,
                   read_runs(probs, aie::load_v<32>(gaps + kb * 32), tables_lo,
                             tables_hi)
                       .pack());
#endif

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

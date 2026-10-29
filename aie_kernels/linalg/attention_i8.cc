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
#include "../common/table_i8.h"
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
// Tables as AIE2P's 4-way LUT read takes them, each 32-byte run twice over
// (the read takes a lane's parity for address bit 5), where the read can reach
// them (the stack it cannot). The exp table an int32 per entry, 8 to a run, so
// a lane's 8-byte read holds its entry's whole word; each of a block's 4
// queries' probabilities an int16 per entry, 16 to a run.
alignas(64) static int32_t exp_runs[512];
alignas(64) static int16_t prob_runs[4 * 512];

// The exp entries of 16 gaps, `offsets` 8 bytes an entry from entry 0. Each
// int32 entry sits in its lane's whole 8-byte read, a layout table_i8.h's
// read_runs (int16 entries, 4 bytes apart) does not take.
static inline __attribute__((always_inline)) aie::vector<uint32, 16>
read_words(const int8_t *base, aie::vector<int32, 16> offsets) {
  v64int8 r0, r1;
  load_lut_2x_int8(base, base, offsets, r0, r1);
  return aie::vector<uint32, 16>((v16uint32)::shuffle(r0, r1, T32_16x2_lo));
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
  const aie::saturation_mode saved_sat =
      aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd =
      aie::swap_rounding(aie::rounding_mode::conv_even);

#if __AIE_ARCH__ >= 21
  for (int r = 0; r < 32; r++) {
    const aie::vector<int32, 8> run = aie::load_v<8>(exp_table + 8 * r);
    aie::store_v(exp_runs + 16 * r, run);
    aie::store_v(exp_runs + 16 * r + 8, run);
  }
  const int8_t *exp_words = reinterpret_cast<const int8_t *>(exp_runs);
  const int8_t *probs = reinterpret_cast<const int8_t *>(prob_runs) + 512;
  // Query q's table starts q * 1024 bytes in.
  alignas(64) static const int32_t query_tables[32] = {
      0,    0,    0,    0,    0,    0,    0,    0,    1024, 1024, 1024,
      1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 2048, 2048,
      2048, 2048, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072};
  // A gap as int8 is gap - 128, its entry's index.
  const aie::vector<int8, 32> flip = aie::broadcast<int8, 32>(-128);
  const aie::vector<int32, 16> tables_lo = aie::load_v<16>(query_tables);
  const aie::vector<int32, 16> tables_hi = aie::load_v<16>(query_tables + 16);
#endif

  alignas(64) int8_t q[KD * W];
  // Scores and probabilities by key block: [N/8][4 queries][8 keys], the
  // layout an mmul A operand reads 8 keys of 4 queries from.
  // Words, so each query's level bounds can reuse them once its block's gaps
  // are taken: the stack has no room for both.
  alignas(64) int32_t s_words[N > 257 ? N : 257];
  int8_t *s = reinterpret_cast<int8_t *>(s_words);
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
    const aie::vector<uint32, 16> low_bits = aie::broadcast<uint32, 16>(0xFFFF);
    aie::vector<uint32, 16> lo_sum0 = aie::zeros<uint32, 16>();
    aie::vector<uint32, 16> lo_sum1 = aie::zeros<uint32, 16>();
    aie::vector<uint32, 16> hi_sum0 = aie::zeros<uint32, 16>();
    aie::vector<uint32, 16> hi_sum1 = aie::zeros<uint32, 16>();
    for (int kb = 0; kb < N / 8; kb++) {
      const aie::vector<uint8, 32> g = aie::load_v<32>(gaps + kb * 32);
      // Entry gap, 8 bytes an entry of offset.
      const aie::vector<int32, 32> wide =
          g.unpack().unpack().template cast_to<int32>();
      const aie::vector<uint32, 16> e0 =
          read_words(exp_words, aie::upshift(wide.template extract<16>(0), 3));
      const aie::vector<uint32, 16> e1 =
          read_words(exp_words, aie::upshift(wide.template extract<16>(1), 3));
      lo_sum0 = aie::add(lo_sum0, aie::bit_and(e0, low_bits));
      lo_sum1 = aie::add(lo_sum1, aie::bit_and(e1, low_bits));
      hi_sum0 = aie::add(hi_sum0, aie::downshift(e0, 16));
      hi_sum1 = aie::add(hi_sum1, aie::downshift(e1, 16));
    }
    alignas(64) uint32_t lo_lanes[32], hi_lanes[32];
    aie::store_v(lo_lanes, lo_sum0);
    aie::store_v(lo_lanes + 16, lo_sum1);
    aie::store_v(hi_lanes, hi_sum0);
    aie::store_v(hi_lanes + 16, hi_sum1);
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
      // The top gap's quotient, exp_table[0] * 2**prob_shift / total, divides
      // by restoring division on total * 2**b: 64-bit division is a libcall
      // Peano lacks. Eight bits are enough because a probability saturates at
      // 127: a quotient of 256 or more takes every bit and rounds to 256.
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
      level += (rem + rem > total) || (rem + rem == total && (level & 1));
      // A gap reaching 127 or more takes 127, so the walk starts there, and
      // 2 * level - 1 below has no bit past steps[7].
      level = level > 127 ? 127 : level;
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
        bar += (((2 * level - 1) >> b) & 1) ? steps[b] : 0;
      // So bar >> k fits 32 bits until it goes below 0, the last level the
      // walk takes, where only its sign counts. It starts below 2**29, so a
      // fall of 2**29 or more takes it there as surely as the whole one, and
      // the walk's two falls from there stay above -2**31. k is at most 31.
      const uint32_t mask = (uint32_t)(unit - 1);
      const int64_t fall = steps[1] >> k;
      const int32_t fall_high = fall > (1 << 29) ? (1 << 29) : (int32_t)fall;
      const uint32_t fall_low = (uint32_t)steps[1] & mask;
      int32_t bar_high = (int32_t)(bar >> k);
      uint32_t bar_low = (uint32_t)bar & mask;
#if __AIE_ARCH__ >= 21
      // Every level's bound first, branch-free; then 16 gaps at a time, each
      // gap's level the highest whose bound its entry exceeds. That is the
      // walk below, without its chain of dependent scalar steps.
      int32_t *bounds = s_words;
      for (int l = level; l > 0; l--) {
        bounds[l] =
            bar_high < 0 ? -1 : bar_high - (bar_low == 0 && (l & 1) == 0);
        // One level down, bar falls by 2 * total.
        bar_high -= fall_high + (bar_low < fall_low);
        bar_low = (bar_low - fall_low) & mask;
      }
      // Run r holds gaps 16r to 16r + 15. The table falls, so once a run's
      // first entry is at most the lowest bound, it and every later run are
      // 0; within a run, levels come down until its last gap has one.
      const aie::vector<int32, 16> none = aie::zeros<int32, 16>();
      int r = 0;
      for (; r < 16 && level > 0 && exp_table[16 * r] > bounds[1]; r++) {
        const aie::vector<int32, 16> entries =
            aie::load_v<16>(exp_table + 16 * r);
        const int32_t last = exp_table[16 * r + 15];
        aie::vector<int32, 16> got = none;
        for (int l = level; l > 0; l--) {
          const int32_t bound = bounds[l];
          const aie::vector<int32, 16> value =
              aie::broadcast<int32, 16>(l > 127 ? 127 : l);
          got = aie::max(
              got,
              aie::select(none, value,
                          aie::gt(entries, aie::broadcast<int32, 16>(bound))));
          if (last > bound)
            break;
        }
        const aie::vector<int16, 16> run = got.pack();
        aie::store_v(prob_runs + i * 512 + 32 * r, run);
        aie::store_v(prob_runs + i * 512 + 32 * r + 16, run);
      }
      for (; r < 16; r++) {
        aie::store_v(prob_runs + i * 512 + 32 * r, aie::zeros<int16, 16>());
        aie::store_v(prob_runs + i * 512 + 32 * r + 16,
                     aie::zeros<int16, 16>());
      }
#else
      alignas(32) int8_t by_gap[256];
      for (int d = 0; d < 256; d += 32)
        aie::store_v(by_gap + d, aie::zeros<int8, 32>());
      for (int d = 0; level > 0 && d < 256; level--) {
        const int32_t bound =
            bar_high < 0 ? -1 : bar_high - (bar_low == 0 && (level & 1) == 0);
        while (d < 256 && exp_table[d] > bound)
          by_gap[d++] = level;
        // One level down, bar falls by 2 * total.
        bar_high -= fall_high + (bar_low < fall_low);
        bar_low = (bar_low - fall_low) & mask;
      }
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
        read[j] = read_runs(
            probs,
            aie::bit_xor(aie::load_v<32>(gaps + (kb + j) * 32).cast_to<int8>(),
                         flip),
            tables_lo, tables_hi);
      AIE_LOOP_UNROLL_FULL
      for (int j = 0; j < GROUP; j++)
        aie::store_v(p + (kb + j) * 32, read[j].pack());
    }
    for (int kb = N / 8 / GROUP * GROUP; kb < N / 8; kb++)
      aie::store_v(
          p + kb * 32,
          read_runs(probs,
                    aie::bit_xor(
                        aie::load_v<32>(gaps + kb * 32).cast_to<int8>(), flip),
                    tables_lo, tables_hi)
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
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}

} // extern "C"

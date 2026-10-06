//===- sample_combine.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Draw one token from SAMPLE_COLUMNS summaries (sample.h), bit for bit as
// aie.iron.kernels.sample.sample_ref does; see it for the definition. Float
// division and subtraction call the soft-float builtins by name: Peano lowers
// `a - b` on floats to the vector unit, which is not IEEE (about 11.5% of
// random operands differ). The helpers stay out of line and the product's
// loops rolled: with the builtins and exp64's table the core's 16 KB of
// program memory is nearly full.

#include <aie_api/aie.hpp>

#include "exp64.h"
#include "sample.h"

#ifndef SAMPLE_COLUMNS
#error                                                                         \
    "SAMPLE_COLUMNS (the columns whose summaries are combined) must be defined"
#endif

extern "C" float __subsf3(float, float);
extern "C" float __divsf3(float, float);

namespace {

float bf16_to_f32(uint32_t bits) {
  return __builtin_bit_cast(float, bits << 16);
}

// The weight of every distinct key >= tau.
struct weights {
  float temperature;
  float xm;
  int32_t tau;
  double w_tau;
  int32_t n;
  int32_t keys[SAMPLE_K_MAX];
  double w[SAMPLE_K_MAX];
};

double weight_of_key(const weights *t, int32_t key) {
  const float xv = __divsf3(bf16_to_f32(sample_bits(key)), t->temperature);
  const float d = __subsf3(xv, t->xm);
  // Above 1 (or NaN) takes a NaN logit or a non-finite max / T, which the
  // contract excludes; clamped to 1 so every sum stays in range.
  const double w = exp64((double)d);
  return __builtin_bit_cast(uint64_t, w) > UINT64_C(0x3ff0000000000000) ? 1.0
                                                                        : w;
}

__attribute__((noinline)) double lookup(weights *t, int32_t key) {
  if (key == t->tau)
    return t->w_tau;
  for (int32_t i = 0; i < t->n; ++i)
    if (t->keys[i] == key)
      return t->w[i];
  const double w = weight_of_key(t, key);
  t->keys[t->n] = key;
  t->w[t->n] = w;
  t->n++;
  return w;
}

// The sums are exact: a float64 weight in [0, 1] is an integer count of
// 2^-1074, and fewer than 2^31 of them (sample.py checks) sum below 2^1105,
// which these words hold, least significant first. The token's prefix P is
// the first with P * 2^53 > n53 * S, i.e. P > q = floor(n53 * S / 2^53):
// S is summed in any order, then each weight added in index order to
// 2^1120 - 1 - q until it carries out. Nothing rounds after the weights.
#define SAMPLE_SUM_WORDS 35

// A weight as m * 2^off counts of 2^-1074: the fields of the double.
struct units {
  uint64_t m;
  int32_t off;
};

units units_of(double w) {
  const uint64_t bits = __builtin_bit_cast(uint64_t, w);
  const int32_t exponent = (int32_t)(bits >> 52);
  const uint64_t fraction = bits & ((UINT64_C(1) << 52) - 1);
  if (exponent == 0)
    return {fraction, 0};
  return {fraction | UINT64_C(1) << 52, exponent - 1};
}

// s += m * 2^off, m < 2^64; whether a carry left the top word.
__attribute__((noinline)) bool add(uint32_t *s, uint64_t m, int32_t off) {
  const int32_t j = off >> 5;
  const int32_t shift = off & 31;
  const uint32_t lo = (uint32_t)m;
  const uint32_t hi = (uint32_t)(m >> 32);
  // x >> (32 - shift) is (x >> 1) >> (31 - shift), defined at shift 0.
  const uint32_t w[3] = {lo << shift, hi << shift | (lo >> 1) >> (31 - shift),
                         (hi >> 1) >> (31 - shift)};
  uint32_t carry = 0;
  for (int32_t i = j; i < SAMPLE_SUM_WORDS; ++i) {
    const uint64_t t = (uint64_t)s[i] + (i < j + 3 ? w[i - j] : 0) + carry;
    s[i] = (uint32_t)t;
    carry = (uint32_t)(t >> 32);
    if (i >= j + 2 && carry == 0)
      return false;
  }
  return carry != 0;
}

// ~q, q = floor(n53 * s / 2^53), n53 < 2^53.
AIE2_MINSIZE __attribute__((noinline)) void
not_scaled(const uint32_t *s, uint64_t n53, uint32_t *not_q) {
  uint32_t product[SAMPLE_SUM_WORDS + 2] = {};
#pragma clang loop unroll(disable)
  for (int32_t half = 0; half < 2; ++half) {
    const uint32_t a = (uint32_t)(n53 >> (32 * half));
    uint64_t carry = 0;
#pragma clang loop unroll(disable)
    for (int32_t i = 0; i < SAMPLE_SUM_WORDS; ++i) {
      const uint64_t t = (uint64_t)a * s[i] + product[i + half] + carry;
      product[i + half] = (uint32_t)t;
      carry = t >> 32;
    }
    product[SAMPLE_SUM_WORDS + half] = (uint32_t)carry;
  }
#pragma clang loop unroll(disable)
  for (int32_t i = 0; i < SAMPLE_SUM_WORDS; ++i)
    not_q[i] = ~(product[i + 1] >> 21 | product[i + 2] << 11);
}

// The row's k-th largest key: every key of the row's top k is in some
// column's summary, so it is the k-th largest over their union (a 256-bin
// rank walk, two bytes deep).
AIE2_MINSIZE __attribute__((noinline)) int32_t
union_tau(const int32_t *summaries, int32_t k) {
  int32_t hist[256];
  int32_t bin = 0;
  int32_t need = k;
  for (int32_t pass = 0; pass < 2; ++pass) {
    for (int32_t b = 0; b < 256; ++b)
      hist[b] = 0;
    for (int32_t c = 0; c < SAMPLE_COLUMNS; ++c) {
      const int32_t *sum = summaries + c * SAMPLE_SUMMARY_WORDS;
      const int32_t n = sum[SAMPLE_ENTRIES];
      for (int32_t e = -1; e < n; ++e) {
        // e == -1 is the column's tau, once per tie.
        const int32_t key =
            e < 0 ? sum[SAMPLE_TAU] : sum[SAMPLE_HEADER + 2 * e + 1];
        const int32_t count = e < 0 ? sum[SAMPLE_TIES] : 1;
        if (pass == 0)
          hist[key >> 8] += count;
        else if ((key >> 8) == bin)
          hist[key & 255] += count;
      }
    }
    const int32_t b = sample_pick(hist, &need);
    bin = pass == 0 ? b : (bin << 8) | b;
  }
  return bin;
}

// S, in any order: each column's entries at or above tau, and w_tau once
// per tie.
AIE2_MINSIZE __attribute__((noinline)) void total(const int32_t *summaries,
                                                  weights *t, uint32_t *s) {
  int32_t ties = 0;
  for (int32_t c = 0; c < SAMPLE_COLUMNS; ++c) {
    const int32_t *sum = summaries + c * SAMPLE_SUMMARY_WORDS;
    const int32_t entries = sum[SAMPLE_ENTRIES];
    for (int32_t e = 0; e < entries; ++e) {
      const int32_t key = sum[SAMPLE_HEADER + 2 * e + 1];
      if (key >= t->tau) {
        const units u = units_of(lookup(t, key));
        add(s, u.m, u.off);
      }
    }
    if (sum[SAMPLE_TAU] == t->tau)
      ties += sum[SAMPLE_TIES];
  }
  // m * ties exactly, in two halves of m.
  const units u = units_of(t->w_tau);
  add(s, (uint64_t)(uint32_t)u.m * (uint32_t)ties, u.off);
  add(s, (u.m >> 32) * (uint32_t)ties, u.off + 32);
}

// Every candidate in index order, each weight added to acc = 2^1120 - 1 - q;
// the first whose prefix P carries acc out of the top word (P > q), or the
// last candidate when none does.
__attribute__((noinline)) int32_t visit(const int32_t *summaries, weights *t,
                                        uint32_t *acc) {
  int32_t last = 0;
  for (int32_t c = 0; c < SAMPLE_COLUMNS; ++c) {
    const int32_t *sum = summaries + c * SAMPLE_SUMMARY_WORDS;
    const int32_t *entry = sum + SAMPLE_HEADER;
    const uint32_t *bitmap = (const uint32_t *)(sum + SAMPLE_BITMAP);
    const int32_t entries = sum[SAMPLE_ENTRIES];
    int32_t ties = sum[SAMPLE_TAU] == t->tau ? sum[SAMPLE_TIES] : 0;
    int32_t e = 0;
    int32_t word = -1;
    uint32_t bits = 0;
    while (e < entries || ties > 0) {
      int32_t tie = SAMPLE_SLICE;
      if (ties > 0) {
        while (bits == 0)
          bits = bitmap[++word];
        tie = word * 32 + sample_ctz(bits);
      }
      int32_t index;
      double w;
      if (e < entries && entry[2 * e] < tie) {
        const int32_t key = entry[2 * e + 1];
        index = entry[2 * e];
        e++;
        if (key < t->tau)
          continue;
        w = lookup(t, key);
      } else {
        index = tie;
        bits &= bits - 1;
        ties--;
        w = t->w_tau;
      }
      last = c * SAMPLE_SLICE + index;
      const units u = units_of(w);
      if (add(acc, u.m, u.off))
        return last;
    }
  }
  return last;
}

int32_t combine(const int32_t *summaries, const int32_t *row) {
  int32_t max_key = -1;
  int32_t argmax = 0;
  for (int32_t c = 0; c < SAMPLE_COLUMNS; ++c) {
    const int32_t *sum = summaries + c * SAMPLE_SUMMARY_WORDS;
    if (sum[SAMPLE_MAX_KEY] > max_key) {
      max_key = sum[SAMPLE_MAX_KEY];
      argmax = c * SAMPLE_SLICE + sum[SAMPLE_ARGMAX];
    }
  }
  const uint32_t temperature_bits = (uint32_t)row[SAMPLE_ROW_TEMPERATURE];
  if ((temperature_bits & 0x7fffffffu) == 0)
    return argmax;

  weights t;
  t.temperature = __builtin_bit_cast(float, temperature_bits);
  t.xm = __divsf3(bf16_to_f32(sample_bits(max_key)), t.temperature);
  t.tau = union_tau(summaries, sample_top_k(row));
  t.w_tau = weight_of_key(&t, t.tau);
  t.n = 0;

  uint32_t s[SAMPLE_SUM_WORDS] = {};
  total(summaries, &t, s);
  const uint64_t n53 = (uint64_t)(uint32_t)row[SAMPLE_ROW_N53_HI] << 32 |
                       (uint32_t)row[SAMPLE_ROW_N53_LO];
  uint32_t acc[SAMPLE_SUM_WORDS];
  not_scaled(s, n53, acc);
  return visit(summaries, &t, acc);
}

} // namespace

extern "C" {

// summaries: SAMPLE_COLUMNS summaries, column 0 first; row: the draw row;
// token and record: the drawn token, twice (the next step's input and the
// position's record).
void sample_combine(const int32_t *summaries, const int32_t *row,
                    int32_t *token, int32_t *record) {
  event0();
  const int32_t drawn = combine(summaries, row);
  *token = drawn;
  *record = drawn;
  event1();
}

} // extern "C"

//===- sample_combine.cc ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Draw one token from SAMPLE_COLUMNS summaries (sample.h), bit for bit as
// aie.iron.kernels.sample.sample_ref does; see it for the definition. The
// prefix sums need their total before any candidate can be judged, so the
// candidates are visited twice. Float division and subtraction call the
// soft-float builtins by name: Peano lowers `a - b` on floats to the vector
// unit, which is not IEEE (about 11.5% of random operands differ).

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
  return exp64((double)d);
}

double lookup(weights *t, int32_t key) {
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

// The running compensated sum and its normalized (h, l).
struct prefix {
  double s, c, h, l;
};

void accumulate(prefix *p, double w) {
  // TwoSum(s, w)
  const double s_next = p->s + w;
  const double bb = s_next - p->s;
  const double a_part = p->s - (s_next - bb);
  const double b_part = w - bb;
  const double e = a_part + b_part;
  p->c = p->c + e;
  p->s = s_next;
  // Fast2Sum(s, c)
  p->h = p->s + p->c;
  const double hs = p->h - p->s;
  p->l = p->c - hs;
}

// Veltkamp: a = *hi + *lo, each half with at most 26 significant bits.
void split(double a, double *hi, double *lo) {
  const double g = 0x1.0000002p27 * a;
  const double r = g - a;
  *hi = g - r;
  *lo = a - *hi;
}

// u * (h + l) as a normalized double-double: TwoProduct(u, h), then + u * l.
void target_of(double u, double h, double l, double *t_hi, double *t_lo) {
  const double p = u * h;
  double u_hi, u_lo, h_hi, h_lo;
  split(u, &u_hi, &u_lo);
  split(h, &h_hi, &h_lo);
  const double e1 = u_hi * h_hi;
  const double e2 = e1 - p;
  const double e3 = u_hi * h_lo;
  const double e4 = e2 + e3;
  const double e5 = u_lo * h_hi;
  const double e6 = e4 + e5;
  const double e7 = u_lo * h_lo;
  const double pe = e6 + e7;
  const double q = u * l;
  const double f = pe + q;
  *t_hi = p + f;
  const double tp = *t_hi - p;
  *t_lo = f - tp;
}

// The row's k-th largest key: every key of the row's top k is in some
// column's summary, so it is the k-th largest over their union (a 256-bin
// rank walk, two bytes deep).
int32_t union_tau(const int32_t *summaries, int32_t k) {
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

// Every candidate in index order into p. With find, stop at the first whose
// prefix exceeds (t_hi, t_lo) and return its index; otherwise, or when none
// does, return the last candidate's index.
int32_t visit(const int32_t *summaries, weights *t, prefix *p, bool find,
              double t_hi, double t_lo) {
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
        tie = word * 32 + __builtin_ctz(bits);
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
      accumulate(p, w);
      if (find && (p->h > t_hi || (p->h == t_hi && p->l > t_lo)))
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

  prefix p = {0.0, 0.0, 0.0, 0.0};
  visit(summaries, &t, &p, false, 0.0, 0.0);
  const uint64_t n53 = (uint64_t)(uint32_t)row[SAMPLE_ROW_N53_HI] << 32 |
                       (uint32_t)row[SAMPLE_ROW_N53_LO];
  const double u = (double)n53 * 0x1p-53;
  double t_hi, t_lo;
  target_of(u, p.h, p.l, &t_hi, &t_lo);
  p = {0.0, 0.0, 0.0, 0.0};
  return visit(summaries, &t, &p, true, t_hi, t_lo);
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

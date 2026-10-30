//===- sample_select.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One column's half of top-k sampling, see sample_combine.cc. The slice
// arrives SAMPLE_SELECT_PASSES times, SAMPLE_CHUNK logits per call, or, when
// a chunk is the whole slice, once, and one call makes every pass. Pass 0
// finds tau, the k-th largest key with multiplicity, and the largest key;
// pass 1 writes keys > tau as (index, key) and keys == tau as bitmap bits.
// The state (a worker-local buffer, zero before the first call) is back where
// it started after the last call, so the next position needs no reset.
//
// SAMPLE_MAX_RANGES and SAMPLE_PROB_RULE (sample.h) add per-call suppress
// ranges and the forced-timestamp rule's second top-k and reduction; see
// apply_ranges and mask_restricted below.

#include <aie_api/aie.hpp>

#include "exp64.h"
#include "sample.h"

#ifndef SAMPLE_CHUNK
#error "SAMPLE_CHUNK (logits per call) must be defined"
#endif

static_assert(SAMPLE_SLICE % SAMPLE_CHUNK == 0,
              "a slice is a whole number of chunks");
static_assert(SAMPLE_SLICE >= SAMPLE_K_MAX,
              "a slice holds at least k logits, so tau exists");

// The slice's logits that are tokens'; the rest pad it to whole words.
#ifndef SAMPLE_VALID
#define SAMPLE_VALID SAMPLE_SLICE
#endif
static_assert(SAMPLE_VALID >= SAMPLE_K_MAX && SAMPLE_VALID <= SAMPLE_SLICE,
              "the valid logits hold at least k and fit the slice");

#define SAMPLE_SELECT_STATE_WORDS 272
#define SAMPLE_SELECT_PASSES 2

namespace {

constexpr int32_t LANES = 32;
constexpr int32_t CANDIDATES = 256;
static_assert(SAMPLE_K_MAX + LANES <= CANDIDATES,
              "a cut buffer has room for one more vector's keys");

using keys_v = aie::vector<int16_t, LANES>;

struct select_state {
  int32_t pos;   // the slice-local index of this call's first logit
  int32_t phase; // 0, 1: the pass
  int32_t k;
  int32_t count; // candidate keys held
  int32_t cut;   // the buffer has been cut to k: thr holds
  int32_t thr;   // pass 0: no key <= thr is needed; pass 1: tau
  int32_t max_key;
  int32_t entries;
  int32_t ties;
  int32_t pad[7];
  int32_t keys[CANDIDATES];
};

static_assert(sizeof(select_state) <= SAMPLE_SELECT_STATE_WORDS * 4,
              "the state fits its buffer");

// sample_key in signed order: -0 is +0, and a negative value's magnitude
// bits are flipped, so larger magnitudes compare lower.
inline keys_v signed_keys(keys_v bits) {
  bits = aie::select(bits, aie::zeros<int16_t, LANES>(),
                     aie::eq(bits, (int16_t)-32768));
  return aie::bit_xor(bits,
                      aie::bit_and((int16_t)0x7fff, aie::downshift(bits, 15)));
}

inline int32_t signed_key(uint16_t bits) {
  return (int32_t)sample_key(bits) - 0x8000;
}

// Reorder a[0..n) so that a[0..k) are its k largest; return the k-th.
int32_t kth_largest(int32_t *a, int32_t n, int32_t k) {
  int32_t lo = 0, hi = n - 1;
  const int32_t t = k - 1;
  while (lo < hi) {
    const int32_t pivot = a[lo + (hi - lo) / 2];
    int32_t i = lo, j = hi;
    while (i <= j) {
      while (a[i] > pivot)
        i++;
      while (a[j] < pivot)
        j--;
      if (i <= j) {
        const int32_t v = a[i];
        a[i++] = a[j];
        a[j--] = v;
      }
    }
    if (t <= j)
      hi = j;
    else if (t >= i)
      lo = i;
    else
      break; // a(j, i) all equal the pivot
  }
  return a[t];
}

// Every key dropped since the last cut is <= thr, and the buffer still holds
// k keys >= thr, so the buffer's k-th largest is the slice's so far.
void cut(select_state *s) {
  s->thr = kth_largest(s->keys, s->count, s->k);
  s->count = s->k;
  s->cut = 1;
}

inline void maybe_cut(select_state *s) {
  if (s->count >= s->k && (!s->cut || s->count > CANDIDATES - LANES))
    cut(s);
}

void begin(select_state *s, const int32_t *row, int32_t *summary) {
  s->k = sample_top_k(row);
  s->count = 0;
  s->cut = 0;
  s->entries = 0;
  s->ties = 0;
  for (int32_t w = 0; w < SAMPLE_BITMAP_WORDS; ++w)
    summary[SAMPLE_BITMAP + w] = 0;
}

void keep(select_state *s, keys_v keys, uint32_t bits) {
  alignas(64) int16_t lane[LANES];
  aie::store_v(lane, keys);
  for (; bits; bits &= bits - 1)
    s->keys[s->count++] = lane[sample_ctz(bits)];
  maybe_cut(s);
}

// Pass 0. The buffer keeps candidate keys, and thr stays no larger than the
// k-th largest key seen so far: a vector with no key above it costs one
// compare, and the buffer is cut back to its k largest when it fills.
void scan(select_state *s, const uint16_t *x) {
  constexpr int32_t vectors = SAMPLE_CHUNK / LANES;
  for (int32_t v = 0; v < vectors; ++v) {
    const keys_v keys =
        signed_keys(aie::load_v<LANES>((const int16_t *)x + v * LANES));
    const uint32_t bits =
        s->cut ? aie::gt(keys, (int16_t)s->thr).to_uint32() : 0xffffffffu;
    if (bits)
      keep(s, keys, bits);
  }
  for (int32_t i = vectors * LANES; i < SAMPLE_CHUNK; ++i) {
    const int32_t key = signed_key(x[i]);
    if (!s->cut || key > s->thr) {
      s->keys[s->count++] = key;
      maybe_cut(s);
    }
  }
}

void tie_bits(int32_t *summary, int32_t index, uint32_t bits) {
  uint32_t *bitmap = (uint32_t *)(summary + SAMPLE_BITMAP);
  const int32_t shift = index & 31;
  bitmap[index >> 5] |= bits << shift;
  if (shift && (bits >> (32 - shift)))
    bitmap[(index >> 5) + 1] |= bits >> (32 - shift);
}

void entry(select_state *s, int32_t *summary, int32_t index, int32_t key) {
  summary[SAMPLE_HEADER + 2 * s->entries] = index;
  summary[SAMPLE_HEADER + 2 * s->entries + 1] = key + 0x8000;
  s->entries++;
}

void collect(select_state *s, const uint16_t *x, int32_t *summary) {
  constexpr int32_t vectors = SAMPLE_CHUNK / LANES;
  const int16_t tau = (int16_t)s->thr;
  for (int32_t v = 0; v < vectors; ++v) {
    const keys_v keys =
        signed_keys(aie::load_v<LANES>((const int16_t *)x + v * LANES));
    const int32_t index = s->pos + v * LANES;
    uint32_t above = aie::gt(keys, tau).to_uint32();
    if (above) {
      alignas(64) int16_t lane[LANES];
      aie::store_v(lane, keys);
      for (; above; above &= above - 1) {
        const int32_t i = sample_ctz(above);
        entry(s, summary, index + i, lane[i]);
      }
    }
    const aie::mask<LANES> equal = aie::eq(keys, tau);
    if (!equal.empty()) {
      tie_bits(summary, index, equal.to_uint32());
#if AIE_HAS_CTZ_POPCOUNT
      s->ties += equal.count();
#else
      // aie::mask::count() is a popcount, which AIE2 cannot lower.
      for (uint32_t bits = equal.to_uint32(); bits; bits &= bits - 1)
        s->ties++;
#endif
    }
  }
  for (int32_t i = vectors * LANES; i < SAMPLE_CHUNK; ++i) {
    const int32_t key = signed_key(x[i]);
    if (key > s->thr)
      entry(s, summary, s->pos + i, key);
    else if (key == s->thr) {
      tie_bits(summary, s->pos + i, 1u);
      s->ties++;
    }
  }
}

// The first index of the largest key: an entry, unless the largest is tau.
int32_t argmax(const select_state *s, const int32_t *summary) {
  if (s->max_key > s->thr) {
    for (int32_t e = 0;; ++e)
      if (summary[SAMPLE_HEADER + 2 * e + 1] == s->max_key + 0x8000)
        return summary[SAMPLE_HEADER + 2 * e];
  }
  const uint32_t *bitmap = (const uint32_t *)(summary + SAMPLE_BITMAP);
  for (int32_t w = 0;; ++w)
    if (bitmap[w])
      return w * 32 + sample_ctz(bitmap[w]);
}

// The end of a pass: what the next one needs.
void end_pass(select_state *s, int32_t *summary) {
  if (s->phase == 0) {
    cut(s);
    int32_t max_key = s->keys[0];
    for (int32_t i = 1; i < s->count; ++i)
      max_key = s->keys[i] > max_key ? s->keys[i] : max_key;
    s->max_key = max_key;
  } else {
    summary[SAMPLE_TAU] = s->thr + 0x8000;
    summary[SAMPLE_ENTRIES] = s->entries;
    summary[SAMPLE_MAX_KEY] = s->max_key + 0x8000;
    summary[SAMPLE_ARGMAX] = argmax(s, summary);
    summary[SAMPLE_TIES] = s->ties;
    for (int32_t w = SAMPLE_TIES + 1; w < SAMPLE_HEADER; ++w)
      summary[w] = 0;
  }
  s->pos = 0;
  s->phase = 1 - s->phase;
}

// A global row coordinate as this column's own, clamped to [0, SAMPLE_SLICE].
// Compared before subtracting: v - SAMPLE_BASE overflows for v near INT32_MIN.
inline int32_t local_coordinate(int32_t v) {
  if (v <= SAMPLE_BASE)
    return 0;
  return v - SAMPLE_BASE > SAMPLE_SLICE ? SAMPLE_SLICE : v - SAMPLE_BASE;
}

#if SAMPLE_MAX_RANGES > 0
// Every SAMPLE_MAX_RANGES (lo, hi) pair, global row coordinates, clipped to
// this call's chunk of this column's slice and masked to SAMPLE_NEG_INF_BITS.
void apply_ranges(uint16_t *x, const int32_t *row, int32_t pos) {
  for (int32_t r = 0; r < SAMPLE_MAX_RANGES; ++r) {
    const int32_t lo = local_coordinate(row[SAMPLE_ROW_RANGES + 2 * r]);
    const int32_t hi = local_coordinate(row[SAMPLE_ROW_RANGES + 2 * r + 1]);
    const int32_t from = lo > pos ? lo : pos;
    const int32_t to = hi < pos + SAMPLE_CHUNK ? hi : pos + SAMPLE_CHUNK;
    for (int32_t i = from; i < to; ++i)
      x[i - pos] = (uint16_t)SAMPLE_NEG_INF_BITS;
  }
}
#endif

#if SAMPLE_PROB_RULE
// This call's chunk, additionally masked below the split, for a second,
// independent top-k (restricted_state/restricted_summary): discarding that
// choice here instead, by excluding entries below the split from the
// ordinary top-k's own surviving candidates, would lose exactly the
// candidates this column's cut() already dropped below its local tau before
// the split was even known. restricted_x is the caller's own
// SAMPLE_CHUNK-wide scratch (too wide for the stack).
void mask_restricted(uint16_t *restricted_x, const uint16_t *x, int32_t pos,
                     int32_t local_split) {
  for (int32_t i = 0; i < SAMPLE_CHUNK; ++i)
    restricted_x[i] = x[i];
  const int32_t to =
      local_split < pos + SAMPLE_CHUNK ? local_split : pos + SAMPLE_CHUNK;
  for (int32_t i = pos; i < to; ++i)
    restricted_x[i - pos] = (uint16_t)SAMPLE_NEG_INF_BITS;
}

void begin_reduction(int32_t *reduction) {
  reduction[SAMPLE_REDUCTION_MAX_TEXT] = (int32_t)0xff800000; // -inf
  reduction[SAMPLE_REDUCTION_MAX_TS] = (int32_t)0xff800000;   // -inf
  reduction[SAMPLE_REDUCTION_SUM_TS] = 0;                     // 0.0f
}

// Folds every logit of this call's chunk (x, already ranges-masked) into
// reduction's running max over its own [0, local_split) part and running
// (max, sum of exp) over its [local_split, SAMPLE_SLICE) part, float32
// throughout; exp64 (float64) evaluates each exp so only the fold back into
// the float32 accumulator rounds.
void reduce_chunk(const uint16_t *x, int32_t pos, int32_t local_split,
                  int32_t *reduction) {
  float max_text =
      __builtin_bit_cast(float, (uint32_t)reduction[SAMPLE_REDUCTION_MAX_TEXT]);
  float max_ts =
      __builtin_bit_cast(float, (uint32_t)reduction[SAMPLE_REDUCTION_MAX_TS]);
  float sum_ts =
      __builtin_bit_cast(float, (uint32_t)reduction[SAMPLE_REDUCTION_SUM_TS]);
  for (int32_t i = 0; i < SAMPLE_CHUNK; ++i) {
    const float v = sample_logit(x[i]);
    if (pos + i < local_split) {
      if (v > max_text)
        max_text = v;
    } else if (v <= max_ts) {
      // -inf weighs exactly 0; folding it before a finite max is -inf - -inf.
      if (x[i] != SAMPLE_NEG_INF_BITS)
        sum_ts = (float)((double)sum_ts + exp64((double)(v - max_ts)));
    } else {
      sum_ts = (float)((double)sum_ts * exp64((double)(max_ts - v)) + 1.0);
      max_ts = v;
    }
  }
  reduction[SAMPLE_REDUCTION_MAX_TEXT] = __builtin_bit_cast(int32_t, max_text);
  reduction[SAMPLE_REDUCTION_MAX_TS] = __builtin_bit_cast(int32_t, max_ts);
  reduction[SAMPLE_REDUCTION_SUM_TS] = __builtin_bit_cast(int32_t, sum_ts);
}
#endif

} // namespace

extern "C" {

// x: SAMPLE_CHUNK bf16 logits; row: the draw row; state: this column's
// state; summary: SAMPLE_SUMMARY_WORDS, the same buffer for all of a
// position's calls. With SAMPLE_PROB_RULE: restricted_state/
// restricted_summary mirror state/summary for the split-restricted top-k;
// restricted_x is this column's own SAMPLE_CHUNK-wide scratch; reduction is
// SAMPLE_REDUCTION_WORDS, this column's partial evidence for the rule's
// compare.
void sample_select(uint16_t *x, const int32_t *row, int32_t *state,
                   int32_t *summary
#if SAMPLE_PROB_RULE
                   ,
                   int32_t *restricted_state, int32_t *restricted_summary,
                   uint16_t *restricted_x, int32_t *reduction
#endif
) {
  event0();
  select_state *s = (select_state *)state;
  // Padding is no token's: -inf, which no draw takes.
  if (SAMPLE_VALID < SAMPLE_SLICE)
    for (int32_t i = SAMPLE_VALID - s->pos; i < SAMPLE_CHUNK; ++i)
      if (i >= 0)
        x[i] = SAMPLE_NEG_INF_BITS;
#if SAMPLE_MAX_RANGES > 0
  apply_ranges(x, row, s->pos);
#endif
#if SAMPLE_PROB_RULE
  select_state *s2 = (select_state *)restricted_state;
  const int32_t local_split = local_coordinate(row[SAMPLE_ROW_SPLIT]);
  mask_restricted(restricted_x, x, s->pos, local_split);
  if (s->phase == 0 && s->pos == 0)
    begin_reduction(reduction);
  if (s->phase == 0)
    reduce_chunk(x, s->pos, local_split, reduction);
#endif
  do {
    if (s->phase == 0 && s->pos == 0)
      begin(s, row, summary);
    if (s->phase == 0)
      scan(s, x);
    else
      collect(s, x, summary);
#if SAMPLE_PROB_RULE
    if (s2->phase == 0 && s2->pos == 0)
      begin(s2, row, restricted_summary);
    if (s2->phase == 0)
      scan(s2, restricted_x);
    else
      collect(s2, restricted_x, restricted_summary);
#endif
    s->pos += SAMPLE_CHUNK;
#if SAMPLE_PROB_RULE
    s2->pos += SAMPLE_CHUNK;
#endif
    if (s->pos == SAMPLE_SLICE) {
      end_pass(s, summary);
#if SAMPLE_PROB_RULE
      end_pass(s2, restricted_summary);
#endif
    }
  } while (SAMPLE_CHUNK == SAMPLE_SLICE && s->phase == 1);
  event1();
}

} // extern "C"

//===- sample_select.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One column's half of top-k sampling, see sample_combine.cc. The slice
// arrives SAMPLE_SELECT_PASSES times, SAMPLE_CHUNK logits per call; or, when
// a chunk is the whole slice, once, and one call makes every pass over it.
// The state (a worker-local buffer, zero before the first call) is back
// where it started after the last call, so the next position needs no reset.
//   pass 0: tau, the k-th largest key with multiplicity, and the largest key
//   pass 1: keys > tau as (index, key), keys == tau as bitmap bits
//
// Pass 0 keeps a buffer of candidate keys and a threshold no larger than the
// k-th largest key seen so far: a vector none of whose keys exceeds it costs
// one compare, and the buffer is cut back to its k largest when it fills.
// Keys are compared as int16 in their signed order, sample_key - 0x8000.

#include <aie_api/aie.hpp>

#include "sample.h"

#ifndef SAMPLE_CHUNK
#error "SAMPLE_CHUNK (logits per call) must be defined"
#endif

static_assert(SAMPLE_SLICE % SAMPLE_CHUNK == 0,
              "a slice is a whole number of chunks");
static_assert(SAMPLE_SLICE >= SAMPLE_K_MAX,
              "a slice holds at least k logits, so tau exists");

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
    s->keys[s->count++] = lane[__builtin_ctz(bits)];
  maybe_cut(s);
}

void scan(select_state *s, const uint16_t *x) {
  constexpr int32_t vectors = SAMPLE_CHUNK / LANES;
  for (int32_t v = 0; v < vectors; ++v) {
    const keys_v keys =
        signed_keys(aie::load_v<LANES>((const int16_t *)x + v * LANES));
    const uint32_t bits = s->cut ? aie::gt(keys, (int16_t)s->thr).to_uint32()
                                 : 0xffffffffu;
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
        const int32_t i = __builtin_ctz(above);
        entry(s, summary, index + i, lane[i]);
      }
    }
    const aie::mask<LANES> equal = aie::eq(keys, tau);
    if (!equal.empty()) {
      tie_bits(summary, index, equal.to_uint32());
      s->ties += equal.count();
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
      return w * 32 + __builtin_ctz(bitmap[w]);
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

} // namespace

extern "C" {

// x: SAMPLE_CHUNK bf16 logits; row: the draw row; state: this column's
// state; summary: SAMPLE_SUMMARY_WORDS, the same buffer for all of a
// position's calls.
void sample_select(const uint16_t *x, const int32_t *row, int32_t *state,
                   int32_t *summary) {
  event0();
  select_state *s = (select_state *)state;
  do {
    if (s->phase == 0 && s->pos == 0)
      begin(s, row, summary);
    if (s->phase == 0)
      scan(s, x);
    else
      collect(s, x, summary);
    s->pos += SAMPLE_CHUNK;
    if (s->pos == SAMPLE_SLICE)
      end_pass(s, summary);
  } while (SAMPLE_CHUNK == SAMPLE_SLICE && s->phase == 1);
  event1();
}

} // extern "C"

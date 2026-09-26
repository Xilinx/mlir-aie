//===- sample_select.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One column's half of top-k sampling, see sample_combine.cc. The slice
// arrives three times, SAMPLE_CHUNK logits per call; the state (a
// worker-local buffer, zero before the first call) is back where it started
// after the last call, so the next position needs no reset.
//   pass 0: high-byte histogram, and the first largest key
//   pass 1: low-byte histogram of the chosen bin: tau
//   pass 2: keys > tau as (index, key), keys == tau as bitmap bits

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

namespace {

struct select_state {
  int32_t hist[256];
  int32_t pos;   // the slice-local index of this call's first logit
  int32_t phase; // 0, 1, 2: the pass
  int32_t k;
  int32_t need; // tau's rank within `bin`
  int32_t bin;  // tau's high byte
  int32_t tau;
  int32_t max_key;
  int32_t argmax;
  int32_t entries;
  int32_t ties;
};

static_assert(sizeof(select_state) <= SAMPLE_SELECT_STATE_WORDS * 4,
              "the state fits its buffer");

void begin(select_state *s, const int32_t *row, int32_t *summary) {
  s->k = sample_top_k(row);
  s->max_key = -1;
  s->argmax = 0;
  s->entries = 0;
  s->ties = 0;
  for (int32_t w = 0; w < SAMPLE_BITMAP_WORDS; ++w)
    summary[SAMPLE_BITMAP + w] = 0;
}

void high_pass(select_state *s, const uint16_t *x) {
  for (int32_t i = 0; i < SAMPLE_CHUNK; ++i) {
    const int32_t key = (int32_t)sample_key(x[i]);
    s->hist[key >> 8]++;
    if (key > s->max_key) {
      s->max_key = key;
      s->argmax = s->pos + i;
    }
  }
}

void low_pass(select_state *s, const uint16_t *x) {
  for (int32_t i = 0; i < SAMPLE_CHUNK; ++i) {
    const int32_t key = (int32_t)sample_key(x[i]);
    if ((key >> 8) == s->bin)
      s->hist[key & 255]++;
  }
}

void collect(select_state *s, const uint16_t *x, int32_t *summary) {
  uint32_t *bitmap = (uint32_t *)(summary + SAMPLE_BITMAP);
  for (int32_t i = 0; i < SAMPLE_CHUNK; ++i) {
    const int32_t key = (int32_t)sample_key(x[i]);
    const int32_t index = s->pos + i;
    if (key > s->tau) {
      summary[SAMPLE_HEADER + 2 * s->entries] = index;
      summary[SAMPLE_HEADER + 2 * s->entries + 1] = key;
      s->entries++;
    } else if (key == s->tau) {
      bitmap[index >> 5] |= 1u << (index & 31);
      s->ties++;
    }
  }
}

void clear_hist(select_state *s) {
  for (int32_t b = 0; b < 256; ++b)
    s->hist[b] = 0;
}

// The end of a pass: what the next one needs.
void end_pass(select_state *s, int32_t *summary) {
  if (s->phase == 0) {
    s->need = s->k;
    s->bin = sample_pick(s->hist, &s->need);
  } else if (s->phase == 1) {
    s->tau = (s->bin << 8) | sample_pick(s->hist, &s->need);
  } else {
    summary[SAMPLE_TAU] = s->tau;
    summary[SAMPLE_ENTRIES] = s->entries;
    summary[SAMPLE_MAX_KEY] = s->max_key;
    summary[SAMPLE_ARGMAX] = s->argmax;
    summary[SAMPLE_TIES] = s->ties;
    for (int32_t w = SAMPLE_TIES + 1; w < SAMPLE_HEADER; ++w)
      summary[w] = 0;
  }
  clear_hist(s);
  s->pos = 0;
  s->phase = s->phase == 2 ? 0 : s->phase + 1;
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
  if (s->phase == 0 && s->pos == 0)
    begin(s, row, summary);
  if (s->phase == 0)
    high_pass(s, x);
  else if (s->phase == 1)
    low_pass(s, x);
  else
    collect(s, x, summary);
  s->pos += SAMPLE_CHUNK;
  if (s->pos == SAMPLE_SLICE)
    end_pass(s, summary);
  event1();
}

} // extern "C"

//===- sample.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// What sample_select.cc and sample_combine.cc agree on: the order key of a
// bf16 logit, the host-written draw row, and a column's summary. The logits
// are split into columns of SAMPLE_SLICE; each select core reduces its slice
// to a summary, and the combine core draws from all of them. Every word is
// int32, and a summary's indices are local to its column.

#ifndef AIE_KERNELS_SAMPLE_SAMPLE_H
#define AIE_KERNELS_SAMPLE_SAMPLE_H

#include <stdint.h>

#include "../aie_arch.h"

#ifndef SAMPLE_SLICE
#error "SAMPLE_SLICE (logits per column) must be defined"
#endif
#ifndef SAMPLE_K_MAX
#error "SAMPLE_K_MAX (the largest top-k) must be defined"
#endif

// Per-call suppress ranges (sample_select only): SAMPLE_MAX_RANGES (lo, hi)
// pairs, global row coordinates, written per call after SAMPLE_DRAW_ROW_WORDS.
// sample_select clips each to its own slice with SAMPLE_BASE, its global
// offset into the row; lo >= hi (after clipping) excludes nothing. 0 is the
// default: no ranges, and sample_select's row is the four words it always
// was.
#ifndef SAMPLE_MAX_RANGES
#define SAMPLE_MAX_RANGES 0
#endif
#ifndef SAMPLE_BASE
#define SAMPLE_BASE 0
#endif
// The forced-timestamp probability rule (sample_select and sample_combine);
// see sample_reduction_word. Off by default, bit-identical to a build that
// predates it.
#ifndef SAMPLE_PROB_RULE
#define SAMPLE_PROB_RULE 0
#endif

// draw_row's own four words; not a SAMPLE_ROW_* name, so
// test_constants_match_the_c_sources's digit-valued scan skips it.
#define SAMPLE_DRAW_ROW_WORDS 4

enum sample_row_word {
  SAMPLE_ROW_TEMPERATURE = 0, // float32 bits; 0 or -0 takes the first argmax
  SAMPLE_ROW_TOP_K = 1,       // clamped to [1, SAMPLE_K_MAX]
  SAMPLE_ROW_N53_LO = 2,      // u = n53 * 2^-53, n53 < 2^53
  SAMPLE_ROW_N53_HI = 3,
  SAMPLE_ROW_RANGES = SAMPLE_DRAW_ROW_WORDS,
  SAMPLE_ROW_SPLIT = SAMPLE_ROW_RANGES + 2 * SAMPLE_MAX_RANGES,
};

// bf16 -inf, what padding and every mask write: it ranks below every finite
// logit, so no masked token is drawn while an unmasked one is finite.
#define SAMPLE_NEG_INF_BITS 0xff80

// sample_select's per-call reduction output (SAMPLE_PROB_RULE only), one
// column's partial evidence for sample_combine's forced-timestamp compare.
// Every word is a float32 bit pattern; SAMPLE_REDUCTION_MAX_TEXT and
// SAMPLE_REDUCTION_MAX_TS are -inf where this column's slice does not reach
// that side of the split, so a column missing one side still merges cleanly.
enum sample_reduction_word {
  SAMPLE_REDUCTION_MAX_TEXT = 0, // max over this slice's [0, split) part
  SAMPLE_REDUCTION_MAX_TS = 1,   // running max over [split, vocab)'s part
  SAMPLE_REDUCTION_SUM_TS = 2,   // sum(exp(v - SAMPLE_REDUCTION_MAX_TS)) there
  SAMPLE_REDUCTION_WORDS = 3,
};

// The bf16 bits of a logit, widened exactly to float (not via a signed key:
// the reduction needs the logit's own value, not its order).
static inline float sample_logit(uint16_t bits) {
  return __builtin_bit_cast(float, (uint32_t)bits << 16);
}

enum sample_summary_word {
  SAMPLE_TAU = 0,     // the column's k-th largest key
  SAMPLE_ENTRIES = 1, // keys > tau, fewer than k
  SAMPLE_MAX_KEY = 2,
  SAMPLE_ARGMAX = 3, // the first index of the largest key
  SAMPLE_TIES = 4,   // keys == tau
  SAMPLE_HEADER = 8, // (index, key) per key > tau, in index order
  // Bit i of word w: index 32 w + i holds tau.
  SAMPLE_BITMAP = SAMPLE_HEADER + 2 * SAMPLE_K_MAX,
  SAMPLE_BITMAP_WORDS = (SAMPLE_SLICE + 31) / 32,
  SAMPLE_SUMMARY_WORDS = SAMPLE_BITMAP + SAMPLE_BITMAP_WORDS,
};

// A uint16 key whose unsigned order is the numeric order of finite bf16
// values and -inf; -0 and +0 are one key.
static inline uint32_t sample_key(uint16_t bits) {
  if (bits == 0x8000)
    bits = 0;
  return (bits & 0x8000) ? (uint16_t)~bits : (uint32_t)(bits | 0x8000);
}

// The bf16 bits of a key (+0 for the zero key).
static inline uint16_t sample_bits(uint32_t key) {
  return (key & 0x8000) ? (uint16_t)(key & 0x7fff) : (uint16_t)~key;
}

// A key's logit, widened exactly to float.
static inline float sample_value(uint32_t key) {
  return __builtin_bit_cast(float, (uint32_t)sample_bits(key) << 16);
}

static inline int32_t sample_top_k(const int32_t *row) {
  int32_t k = row[SAMPLE_ROW_TOP_K];
  if (k < 1)
    return 1;
  return k > SAMPLE_K_MAX ? SAMPLE_K_MAX : k;
}

// The bin of a 256-bin histogram that holds the need-th largest element,
// counting from bin 255 down; need becomes its rank within that bin.
static inline int32_t sample_pick(const int32_t *hist, int32_t *need) {
  for (int32_t b = 255; b > 0; --b) {
    if (hist[b] >= *need)
      return b;
    *need -= hist[b];
  }
  return 0;
}

// The trailing zeros of a nonzero word. AIE2 has no lowering for ctz, so
// there it isolates the lowest set bit and counts the zeros above it.
static inline int32_t sample_ctz(uint32_t x) {
#if AIE_HAS_CTZ_POPCOUNT
  return __builtin_ctz(x);
#else
  return 31 - __builtin_clz(x & (0u - x));
#endif
}

#endif

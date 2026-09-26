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

#ifndef SAMPLE_SLICE
#error "SAMPLE_SLICE (logits per column) must be defined"
#endif
#ifndef SAMPLE_K_MAX
#error "SAMPLE_K_MAX (the largest top-k) must be defined"
#endif

enum sample_row_word {
  SAMPLE_ROW_TEMPERATURE = 0, // float32 bits; 0 or -0 takes the first argmax
  SAMPLE_ROW_TOP_K = 1,       // clamped to [1, SAMPLE_K_MAX]
  SAMPLE_ROW_N53_LO = 2,      // u = n53 * 2^-53, n53 < 2^53
  SAMPLE_ROW_N53_HI = 3,
};

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

#endif

//===- llama_topk_set.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The lm_head's top-k sets (llama_mma_lm_topk per core, llama_lmhead_merge
// across them): unsorted, ordered by logit, then by lower gidx. Append until
// full, then replace the lowest member when a higher one arrives, which
// leaves the top kKset under that order whatever the insertion order
// (llama_oracle.topk_set_reference).
//===----------------------------------------------------------------------===//

#ifndef LLAMA_TOPK_SET_H
#define LLAMA_TOPK_SET_H

#include <stdint.h>

// A signed key ordered as IEEE orders non-NaN floats, with -0 tying +0.
static inline int32_t topk_key(float f) {
  int32_t b = __builtin_bit_cast(int32_t, f);
  if (b == INT32_MIN)
    b = 0;
  return b ^ ((b >> 31) & 0x7fffffff);
}

// A set's insertion state over len members; key_of(i) and gidx_of(i) read
// member i, so the caller keeps the members in its own layout.
template <int kKset>
struct TopkInsert {
  int len;
  bool rescan = true;
  int mi = 0;
  int32_t min_key = 0;
  int32_t min_gidx = 0;

  // The slot candidate (key, gidx) goes to, or -1 when the set keeps out.
  template <typename KeyOf, typename GidxOf>
  int slot(int32_t key, int32_t gidx, KeyOf key_of, GidxOf gidx_of) {
    if (len < kKset)
      return len++;
    if (rescan) {
      mi = 0;
      min_key = key_of(0);
      min_gidx = gidx_of(0);
      for (int i = 1; i < kKset; i++) {
        const int32_t k = key_of(i);
        const int32_t g = gidx_of(i);
        if (k < min_key || (k == min_key && g > min_gidx)) {
          min_key = k;
          min_gidx = g;
          mi = i;
        }
      }
      rescan = false;
    }
    if (key < min_key || (key == min_key && gidx > min_gidx))
      return -1;
    rescan = true;
    return mi;
  }
};

#endif // LLAMA_TOPK_SET_H

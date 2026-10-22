//===- llama_lmhead_topk.cc ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The lm_head's resident top-k set, merged from the GEMV array's per-core sets.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr int kK = 2048, kRows = 4;

// A signed key ordered as IEEE orders non-NaN floats, with -0 tying +0.
static inline int32_t logit_key(float f) {
  int32_t b = __builtin_bit_cast(int32_t, f);
  if (b == INT32_MIN)
    b = 0;
  return b ^ ((b >> 31) & 0x7fffffff);
}

// Each GEMV core keeps the top of its share of the vocabulary with their
// embed rows (llama_mma_lm_topk); a candidate slot is half of one core's set:
// 4 rows, then 16-byte blocks of their logits, gidx and scales and the core
// set's length. The union of the cores' sets holds the vocabulary's top
// kKset, so inserting every candidate gives the set one pass over the table
// would. The set is unsorted, ordered by logit, then by lower gidx: append
// until full, then replace the lowest member when a higher one arrives,
// which leaves the top kKset under that order whatever the insertion order
// (llama_oracle.topk_set_reference). llama_topk_finalize samples it; its
// layout and kKset must match.
template <int kKset>
static inline void
lmhead_merge_impl(int8_t *restrict cand, float *restrict set_logit,
                  int32_t *restrict set_gidx, float *restrict set_scale,
                  int8_t *restrict set_row, int32_t *restrict set_len,
                  int32_t half, int32_t first) {
  if (first)
    *set_len = 0;
  const int32_t *meta = reinterpret_cast<const int32_t *>(cand + kRows * kK);
  const int core_len = meta[12];
  int len = *set_len;
  bool rescan = true;
  int mi = 0;
  int32_t min_key = 0;
  int32_t min_gidx = 0;
  for (int n = 0; n < kRows && half * kRows + n < core_len; n++) {
    const float logit = __builtin_bit_cast(float, meta[n]);
    const int32_t gidx = meta[kRows + n];
    int slot;
    if (len < kKset) {
      slot = len++;
    } else {
      if (rescan) {
        mi = 0;
        min_key = logit_key(set_logit[0]);
        min_gidx = set_gidx[0];
        for (int i = 1; i < kKset; i++) {
          int32_t k = logit_key(set_logit[i]);
          if (k < min_key || (k == min_key && set_gidx[i] > min_gidx)) {
            min_key = k;
            min_gidx = set_gidx[i];
            mi = i;
          }
        }
        rescan = false;
      }
      int32_t key = logit_key(logit);
      if (key < min_key || (key == min_key && gidx > min_gidx))
        continue;
      slot = mi;
      rescan = true;
    }
    set_logit[slot] = logit;
    set_gidx[slot] = gidx;
    set_scale[slot] = __builtin_bit_cast(float, meta[2 * kRows + n]);
    for (int c = 0; c < kK / 64; c++)
      ::aie::store_v(set_row + slot * kK + c * 64,
                     ::aie::load_v<64>(cand + n * kK + c * 64));
  }
  *set_len = len;
}

#ifndef LLAMA_LMHEAD_KSET
#define LLAMA_LMHEAD_KSET 16
#endif

extern "C" {

void llama_lmhead_merge(int8_t *restrict cand, float *restrict set_logit,
                        int32_t *restrict set_gidx, float *restrict set_scale,
                        int8_t *restrict set_row, int32_t *restrict set_len,
                        int32_t half, int32_t first) {
  lmhead_merge_impl<LLAMA_LMHEAD_KSET>(cand, set_logit, set_gidx, set_scale,
                                       set_row, set_len, half, first);
}

} // extern "C"

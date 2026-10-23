//===- llama_lmhead_topk.cc ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The lm_head's top-k set, merged from the GEMV array's per-core sets.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_topk_set.h"

#ifndef LLAMA_LMHEAD_KSET
#define LLAMA_LMHEAD_KSET 8
#endif

static constexpr int kK = 2048, kRows = 4;
static constexpr int kKset = LLAMA_LMHEAD_KSET;

extern "C" {

// Each GEMV core keeps the top of its share of the vocabulary with their
// embed rows (llama_mma_lm_topk); a candidate slot is half of one core's set:
// 4 rows, then 16-byte blocks of their logits, gidx and scales and the core
// set's length. The union of the cores' sets holds the vocabulary's top
// kKset, so inserting every candidate gives the set that one pass over the
// table would. llama_topk_finalize samples it; its layout and kKset must
// match.
void llama_lmhead_merge(int8_t *restrict cand, float *restrict set_logit,
                        int32_t *restrict set_gidx, float *restrict set_scale,
                        int8_t *restrict set_row, int32_t *restrict set_len,
                        int32_t half, int32_t first) {
  event0();
  if (first)
    *set_len = 0;
  const int32_t *meta = reinterpret_cast<const int32_t *>(cand + kRows * kK);
  const int core_len = meta[12];
  TopkInsert<kKset> set{*set_len};
  for (int n = 0; n < kRows && half * kRows + n < core_len; n++) {
    const float logit = __builtin_bit_cast(float, meta[n]);
    const int32_t gidx = meta[kRows + n];
    const int slot = set.slot(
        topk_key(logit), gidx, [&](int i) { return topk_key(set_logit[i]); },
        [&](int i) { return set_gidx[i]; });
    if (slot < 0)
      continue;
    set_logit[slot] = logit;
    set_gidx[slot] = gidx;
    set_scale[slot] = __builtin_bit_cast(float, meta[2 * kRows + n]);
    for (int c = 0; c < kK / 64; c++)
      ::aie::store_v(set_row + slot * kK + c * 64,
                     ::aie::load_v<64>(cand + n * kK + c * 64));
  }
  *set_len = set.len;
  event1();
}

} // extern "C"

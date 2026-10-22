//===- llama_lmhead_topk.cc ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The lm_head as a fused GEMV and top-k insert over the tied embed table.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_vec_fp.h"

static constexpr int kK = 2048, kNTile = 4, kPrefix = 64, kVec = 64;

// A signed key ordered as IEEE orders non-NaN floats, with -0 tying +0.
static inline int32_t logit_key(float f) {
  int32_t b = __builtin_bit_cast(int32_t, f);
  if (b == INT32_MIN)
    b = 0;
  return b ^ ((b >> 31) & 0x7fffffff);
}

// A weight slot's kNTile int32 sums against act, its biases added.
static inline void tile_sums(const int8_t *restrict act,
                             const int8_t *restrict w_tile, int32_t *sums) {
  const int8_t *weights = w_tile + kPrefix;
  const int32_t *bias =
      reinterpret_cast<const int32_t *>(weights + kNTile * kK);
  static_assert(kNTile == 4);
  aie::accum<acc32, kVec> acc0, acc1, acc2, acc3;
  acc0.from_vector(aie::zeros<int32, kVec>());
  acc1.from_vector(aie::zeros<int32, kVec>());
  acc2.from_vector(aie::zeros<int32, kVec>());
  acc3.from_vector(aie::zeros<int32, kVec>());
  const int8_t *__restrict w0 = weights;
  const int8_t *__restrict w1 = weights + kK;
  const int8_t *__restrict w2 = weights + 2 * kK;
  const int8_t *__restrict w3 = weights + 3 * kK;
  const int8_t *__restrict x = act;
#pragma clang loop unroll_count(2)
  for (int g = 0; g < kK / kVec; g++) {
    aie::vector<int8, kVec> x_v = aie::load_v<kVec>(x);
    acc0 = aie::mac(acc0, aie::load_v<kVec>(w0), x_v);
    acc1 = aie::mac(acc1, aie::load_v<kVec>(w1), x_v);
    acc2 = aie::mac(acc2, aie::load_v<kVec>(w2), x_v);
    acc3 = aie::mac(acc3, aie::load_v<kVec>(w3), x_v);
    x += kVec;
    w0 += kVec;
    w1 += kVec;
    w2 += kVec;
    w3 += kVec;
  }
  sums[0] = aie::reduce_add(acc0.template to_vector<int32>()) + bias[0];
  sums[1] = aie::reduce_add(acc1.template to_vector<int32>()) + bias[1];
  sums[2] = aie::reduce_add(acc2.template to_vector<int32>()) + bias[2];
  sums[3] = aie::reduce_add(acc3.template to_vector<int32>()) + bias[3];
}

// The lm_head's logits for one slot of each lane (tiles tile_a and tile_b),
// inserted with their global index, embed_sc and embed row into a resident
// top-k set instead of written to DDR: tied embeddings make the GEMV's weight
// row the embed row, so one table pass needs no second gather.
// llama_topk_finalize samples the set; its layout and kKset must match. The
// set is unsorted, ordered by logit, then by lower gidx: append until full,
// then replace the lowest member when a higher one arrives, which leaves the
// vocabulary's top kKset under that order whatever the insertion order
// (llama_oracle.topk_set_reference).
template <int kKset>
static inline void
lmhead_topk_insert_impl(int8_t *restrict act, int8_t *restrict w_a,
                        int8_t *restrict w_b, float *restrict set_logit,
                        int32_t *restrict set_gidx, float *restrict set_scale,
                        int8_t *restrict set_row, int32_t *restrict set_len,
                        int32_t tile_a, int32_t tile_b) {
  constexpr int kRows = 2 * kNTile;
  int8_t *const tiles[2] = {w_a, w_b};
  const int32_t tile_idx[2] = {tile_a, tile_b};

  // Tile 0 starts a table pass: the set persists across passes and
  // dispatches, so stale entries would leak in.
  if (tile_a == 0)
    *set_len = 0;

  float act_scale = *reinterpret_cast<const float *>(act + kK);
  float w_scales[kRows];
  for (int s = 0; s < 2; s++)
    for (int n = 0; n < kNTile; n++)
      w_scales[s * kNTile + n] = reinterpret_cast<const float *>(
          tiles[s] + kPrefix + kNTile * kK + kNTile * 4)[n];

  const aie::saturation_mode saved_sat =
      ::aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  int32_t sums[kRows];
  tile_sums(act, w_a, sums);
  tile_sums(act, w_b, sums + kNTile);

  float logits[kRows];
  if (vec_epilogue_exact<80, 174, kRows>(sums, act_scale, w_scales)) {
    ::aie::vector<float, 16> logit_v =
        vec_scaled_sums<kRows>(sums, act_scale, w_scales);
    for (int n = 0; n < kRows; n++)
      logits[n] = logit_v.get(n);
  } else {
    for (int n = 0; n < kRows; n++)
      logits[n] = (float)sums[n] * act_scale * w_scales[n];
  }

  int len = *set_len;
  if (len == kKset) {
    // Almost every call inserts nothing: compare the call's best key with the
    // set's lowest key (one vector pass) before the scan that breaks ties.
    ::aie::vector<int32, kKset> b =
        ::aie::load_unaligned_v<kKset>(set_logit).template cast_to<int32>();
    b = ::aie::select(b, ::aie::zeros<int32, kKset>(),
                      ::aie::eq(b, int32(INT32_MIN)));
    int32_t set_min = ::aie::reduce_min(::aie::bit_xor(
        b, ::aie::bit_and(int32(0x7fffffff), ::aie::downshift(b, 31))));
    int32_t best = logit_key(logits[0]);
    for (int n = 1; n < kRows; n++) {
      int32_t k = logit_key(logits[n]);
      best = k > best ? k : best;
    }
    if (best < set_min) {
      aie::set_rounding(saved_rnd);
      aie::set_saturation(saved_sat);
      return;
    }
  }
  bool rescan = true;
  int mi = 0;
  int32_t min_key = 0;
  int32_t min_gidx = 0;
  for (int n = 0; n < kRows; n++) {
    const int s = n / kNTile, r = n % kNTile;
    const int32_t gidx = tile_idx[s] * kNTile + r;
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
      int32_t key = logit_key(logits[n]);
      if (key < min_key || (key == min_key && gidx > min_gidx))
        continue;
      slot = mi;
      rescan = true;
    }
    const int8_t *__restrict w_row = tiles[s] + kPrefix + r * kK;
    set_logit[slot] = logits[n];
    set_gidx[slot] = gidx;
    set_scale[slot] = w_scales[n];
    for (int d = 0; d < kK; d++)
      set_row[slot * kK + d] = w_row[d];
  }
  *set_len = len;
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
}

#ifndef LLAMA_LMHEAD_KSET
#define LLAMA_LMHEAD_KSET 16
#endif

extern "C" {

void llama_lmhead_topk_insert(int8_t *restrict act, int8_t *restrict w_a,
                              int8_t *restrict w_b, float *restrict set_logit,
                              int32_t *restrict set_gidx,
                              float *restrict set_scale,
                              int8_t *restrict set_row,
                              int32_t *restrict set_len, int32_t tile_a,
                              int32_t tile_b) {
  event0();
  lmhead_topk_insert_impl<LLAMA_LMHEAD_KSET>(act, w_a, w_b, set_logit, set_gidx,
                                             set_scale, set_row, set_len,
                                             tile_a, tile_b);
  event1();
}

// The final norm's output from the head of a weight slot.
void llama_lmhead_take_act(int8_t *restrict slot, int8_t *restrict act) {
  copy_words<kK + 16>(act, slot);
}

} // extern "C"

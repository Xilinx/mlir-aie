//===- llama_gemv_mma.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The GEMV array's cores on aie::mmul<8, 8, 8>: eight activations (A) by
// eight weight rows (B). Activations: a[m][8kt + i] at kt*64 + m*8 + i
// (8 contiguous bytes, so a DMA can gather them), then 8 act_scales. Slot:
// half of K for 8 rows, W[n][8kt + i] at kt*64 + i*8 + n, then 8 biases and
// 8 w_scales.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"
#include "llama_topk_set.h"

static constexpr int kHalfTiles = 1024 / 8;
static constexpr int kScalesOff = 8 * 2048;
static constexpr int kTrailer = 8 * 1024;
using MMUL = ::aie::mmul<8, 8, 8, int8, int8, acc32>;

// dst[m*8 + n] = weight row n against activation m over one half of K,
// plus add[m*8 + n] unless add is null. Out of line, as is half_epilogue,
// so the entry points share one copy in the 16 KB program memory.
__attribute__((noinline)) static void half_sums(const int8_t *restrict act,
                                                const int8_t *restrict w,
                                                int half, int32_t *dst,
                                                const int32_t *add) {
  const int8_t *a = act + half * kHalfTiles * 64;
  MMUL c;
  c.mul(::aie::load_v<64>(a), ::aie::load_v<64>(w));
  for (int kt = 1; kt < kHalfTiles; kt++)
    c.mac(::aie::load_v<64>(a + kt * 64), ::aie::load_v<64>(w + kt * 64));
  ::aie::accum<acc32, 64> acc = c.to_accum();
  for (int i = 0; i < 2; i++) {
    ::aie::vector<int32, 32> v =
        acc.template extract<32>(i).template to_vector<int32>();
    if (add)
      v = ::aie::add(v, ::aie::load_v<32>(add + 32 * i));
    ::aie::store_v(dst + 32 * i, v);
  }
}

static inline void epilogue(const int8_t *restrict act,
                            const int8_t *restrict w,
                            const int32_t *restrict part, int8_t *restrict out,
                            int32_t half, int32_t ostride, int32_t row0,
                            int32_t m_count) {
  alignas(64) int32_t sums[64];
  half_sums(act, w, half, sums, part);
  const aie::saturation_mode saved_sat =
      ::aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  const float *act_scales = reinterpret_cast<const float *>(act + kScalesOff);
  half_epilogue(::aie::load_v<32>(sums), act_scales, w + kTrailer, out, 0,
                ostride, row0, m_count);
  if (m_count > 4)
    half_epilogue(::aie::load_v<32>(sums + 32), act_scales + 4, w + kTrailer,
                  out, 1, ostride, row0, m_count);
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
}

extern "C" {

// The mma tile from m_count (at most 4) activations of src: activation m's
// columns k_at..k_at + 2048 at src + m*row_stride + k_at in tile slot m, zeros
// in the slots past m_count (their sums are bias only, inside the epilogue's
// exact range); activation m's scale at src + scales_at + m*scale_stride, 1.0
// for the others.
void llama_mma_spread(int8_t *restrict src, int8_t *restrict dst,
                      int32_t m_count, int32_t row_stride, int32_t k_at,
                      int32_t scales_at, int32_t scale_stride) {
  event0();
  const ::aie::vector<int8, 32> zero = ::aie::zeros<int8, 32>();
  for (int c = 0; c < 2048 / 64; c++) {
    ::aie::vector<int8, 64> v[4];
    for (int m = 0; m < 4; m++)
      v[m] = m < m_count ? ::aie::load_unaligned_v<64>(src + m * row_stride +
                                                       k_at + c * 64)
                         : ::aie::zeros<int8, 64>();
    // Tile kt's slots 0-3: the four rows' bytes 8kt..8kt + 7, in turn.
    auto [a0, a1] = ::aie::interleave_zip(v[0], v[1], 8);
    auto [b0, b1] = ::aie::interleave_zip(v[2], v[3], 8);
    auto [t0, t1] = ::aie::interleave_zip(a0, b0, 16);
    auto [t2, t3] = ::aie::interleave_zip(a1, b1, 16);
    int8_t *d = dst + c * 8 * 64;
    ::aie::store_v(d, ::aie::concat(t0.extract<32>(0), zero));
    ::aie::store_v(d + 64, ::aie::concat(t0.extract<32>(1), zero));
    ::aie::store_v(d + 128, ::aie::concat(t1.extract<32>(0), zero));
    ::aie::store_v(d + 192, ::aie::concat(t1.extract<32>(1), zero));
    ::aie::store_v(d + 256, ::aie::concat(t2.extract<32>(0), zero));
    ::aie::store_v(d + 320, ::aie::concat(t2.extract<32>(1), zero));
    ::aie::store_v(d + 384, ::aie::concat(t3.extract<32>(0), zero));
    ::aie::store_v(d + 448, ::aie::concat(t3.extract<32>(1), zero));
  }
  float *scales = reinterpret_cast<float *>(dst + kScalesOff);
  for (int m = 0; m < 8; m++)
    scales[m] = m < m_count ? *reinterpret_cast<const float *>(src + scales_at +
                                                               m * scale_stride)
                            : 1.0f;
  event1();
}

// Partial sums group g (int32[64] at part + 64g, lane m*8 + n) = (first) or
// += this slot's sums against the activations' half `half` of K.
void llama_mma_acc(int8_t *restrict act, int8_t *restrict w,
                   int32_t *restrict part, int32_t g, int32_t half,
                   int32_t first) {
  event0();
  half_sums(act, w, half, part + 64 * g, first ? nullptr : part + 64 * g);
  event1();
}

// Group g's partial sums plus this slot's, through the f32 epilogue with
// this slot's bias and w_scales: out (fp32) [m*ostride + row0 + n] for
// m < m_count.
void llama_mma_f32(int8_t *restrict act, int8_t *restrict w,
                   int32_t *restrict part, int8_t *restrict out, int32_t g,
                   int32_t half, int32_t ostride, int32_t row0,
                   int32_t m_count) {
  event0();
  epilogue(act, w, part + 64 * g, out, half, ostride, row0, m_count);
  event1();
}

// llama_mma_f32 under a second symbol for a second output type; an alias
// shares the one body.
void llama_mma_f32_to(int8_t *restrict act, int8_t *restrict w,
                      int32_t *restrict part, int8_t *restrict out, int32_t g,
                      int32_t half, int32_t ostride, int32_t row0,
                      int32_t m_count) __attribute__((alias("llama_mma_f32")));

} // extern "C"

// The lm_head on the array. A table slot is 4 rows of the tied embed table,
// row-major (kLmRows x 2048), then 8 biases and 8 w_scales as a weight slot's
// trailer (rows 4-7: bias 0, scale 1). Each core keeps the top kLmKset of its
// share of the vocabulary by (logit, lower gidx), with their embed rows (in
// the idle mma tile) and their meta (in the idle partials), the sampler core's
// candidate layout: per 4 entries kMetaBlock words, their logits, gidx and
// scales, then the set's length.
#ifndef LLAMA_LM_KSET
#define LLAMA_LM_KSET 8
#endif
#ifndef LLAMA_GEMV_M_MAX
#define LLAMA_GEMV_M_MAX 4
#endif
#ifndef LLAMA_GEMV_OUT_BYTES
#define LLAMA_GEMV_OUT_BYTES 4096
#endif
static constexpr int kLmKset = LLAMA_LM_KSET;
static constexpr int kLmRows = 4;
static constexpr int kD = 2048;
static constexpr int kRowsPerOut = LLAMA_GEMV_OUT_BYTES / kD;
static constexpr int kMetaBlock = 16;
static constexpr int kMetaLen = 12;
static_assert(kLmKset == 8, "the set is the mma tile's 8 rows, 2 meta blocks");
static_assert(LLAMA_GEMV_OUT_BYTES % kD == 0 && kLmKset % kRowsPerOut == 0,
              "an out object holds whole rows of the set");

extern "C" {

// The logits of table slot `tile_idx` against act's first activation (a
// decode activation object: row 0's 2048 columns, its scale after the
// LLAMA_GEMV_M_MAX rows), inserted into the core's set. Tile 0 of the core's
// share resets the set.
void llama_mma_lm_topk(int8_t *restrict act, int8_t *restrict w,
                       int8_t *restrict rows, int32_t *restrict meta,
                       int32_t tile_idx, int32_t first) {
  event0();
  if (first)
    meta[kMetaLen] = 0;

  aie::accum<acc32, 64> acc0, acc1, acc2, acc3;
  acc0.from_vector(aie::zeros<int32, 64>());
  acc1.from_vector(aie::zeros<int32, 64>());
  acc2.from_vector(aie::zeros<int32, 64>());
  acc3.from_vector(aie::zeros<int32, 64>());
  const int8_t *__restrict w0 = w;
  const int8_t *__restrict w1 = w + kD;
  const int8_t *__restrict w2 = w + 2 * kD;
  const int8_t *__restrict w3 = w + 3 * kD;
  const int8_t *__restrict x = act;
#pragma clang loop unroll_count(2)
  for (int g = 0; g < kD / 64; g++) {
    aie::vector<int8, 64> x_v = aie::load_v<64>(x);
    acc0 = aie::mac(acc0, aie::load_v<64>(w0), x_v);
    acc1 = aie::mac(acc1, aie::load_v<64>(w1), x_v);
    acc2 = aie::mac(acc2, aie::load_v<64>(w2), x_v);
    acc3 = aie::mac(acc3, aie::load_v<64>(w3), x_v);
    x += 64;
    w0 += 64;
    w1 += 64;
    w2 += 64;
    w3 += 64;
  }
  ::aie::vector<int32, 32> sums = ::aie::zeros<int32, 32>();
  sums.set(aie::reduce_add(acc0.template to_vector<int32>()), 0);
  sums.set(aie::reduce_add(acc1.template to_vector<int32>()), 1);
  sums.set(aie::reduce_add(acc2.template to_vector<int32>()), 2);
  sums.set(aie::reduce_add(acc3.template to_vector<int32>()), 3);

  alignas(16) float act_scales[4] = {
      *reinterpret_cast<const float *>(act + LLAMA_GEMV_M_MAX * kD), 1.0f, 1.0f,
      1.0f};
  alignas(32) float logits[8];
  const aie::saturation_mode saved_sat =
      ::aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  half_epilogue(sums, act_scales, w + kTrailer,
                reinterpret_cast<int8_t *>(logits), 0, 0, 0, 1);
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);

  const float *w_scales = reinterpret_cast<const float *>(w + kTrailer + 32);
  TopkInsert<kLmKset> set{meta[kMetaLen]};
  for (int n = 0; n < kLmRows; n++) {
    const int32_t gidx = tile_idx * kLmRows + n;
    const int slot = set.slot(
        topk_key(logits[n]), gidx,
        [&](int i) {
          return topk_key(
              __builtin_bit_cast(float, meta[kMetaBlock * (i / 4) + i % 4]));
        },
        [&](int i) { return meta[kMetaBlock * (i / 4) + i % 4 + 4]; });
    if (slot < 0)
      continue;
    const int at = kMetaBlock * (slot / 4) + slot % 4;
    meta[at] = __builtin_bit_cast(int32_t, logits[n]);
    meta[at + 4] = gidx;
    meta[at + 8] = __builtin_bit_cast(int32_t, w_scales[n]);
    for (int c = 0; c < kD / 64; c++)
      ::aie::store_v(rows + slot * kD + c * 64,
                     ::aie::load_v<64>(w + n * kD + c * 64));
  }
  meta[kMetaLen] = set.len;
  meta[kMetaBlock + kMetaLen] = set.len;
  event1();
}

// Out object j of the core's set for the sampler core: below
// kLmKset / kRowsPerOut its rows from kRowsPerOut * j, then its meta (the
// partials' first LLAMA_GEMV_OUT_BYTES bytes).
void llama_mma_lm_out(int8_t *restrict rows, int32_t *restrict meta,
                      int8_t *restrict out, int32_t j) {
  event0();
  const int8_t *src = j < kLmKset / kRowsPerOut
                          ? rows + kRowsPerOut * j * kD
                          : reinterpret_cast<const int8_t *>(meta);
  for (int c = 0; c < LLAMA_GEMV_OUT_BYTES / 64; c++)
    ::aie::store_v(out + c * 64, ::aie::load_v<64>(src + c * 64));
  event1();
}

} // extern "C"

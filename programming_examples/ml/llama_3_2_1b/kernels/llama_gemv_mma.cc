//===- llama_gemv_mma.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The GEMV array's cores on aie::mmul<8, 8, 8>: eight activations (A) by
// eight weight rows (B). Activations: a[m][8kt + i] at kt*64 + m*8 + i
// (8 contiguous bytes, so a DMA can gather them), then 8 act_scales. Slot:
// half of K for 8 rows, W[n][8kt + i] at kt*64 + i*8 + n, then 8 biases and
// 8 w_scales. Sums and epilogue are the GEMV's, so rows are byte-identical.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"

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

// The f32 epilogue on activations 4h..4h+3's 32 sums (lane (m - 4h)*8 + n,
// act_scales[m - 4h]), bias added, written to out[m*ostride + row0 + n] for
// the first m_count activations: the GEMV epilogue's arithmetic per element,
// on all 32 lanes at once where the GEMV's guards hold for every lane (see
// vec_epilogue_exact), else its scalar code per element.
__attribute__((noinline)) static void
half_epilogue(::aie::vector<int32, 32> sums, const float *restrict act_scales,
              const int8_t *restrict w, int8_t *restrict out, int h,
              int32_t ostride, int32_t row0, int32_t m_count) {
  const int32_t *bias = reinterpret_cast<const int32_t *>(w + kTrailer);
  const float *w_scales = reinterpret_cast<const float *>(w + kTrailer + 32);
  const int m_end = m_count - 4 * h < 4 ? m_count - 4 * h : 4;
  const uint32_t kLo = 80, kHi = 174;
  bool ok = true;
  for (int i = 0; i < 4 + 8; i++) {
    float f = i < 4 ? act_scales[i] : w_scales[i - 4];
    ok &= (uint32_t)(((__builtin_bit_cast(int32_t, f) >> 23) & 0xff) - kLo) <=
          kHi - kLo;
  }
  ::aie::vector<float, 8> w8 = ::aie::load_v<8>(w_scales);
  ::aie::vector<float, 32> w_v = ::aie::concat(w8, w8, w8, w8);
  ::aie::vector<float, 32> a_v =
      ::aie::concat(::aie::broadcast<float, 8>(act_scales[0]),
                    ::aie::broadcast<float, 8>(act_scales[1]),
                    ::aie::broadcast<float, 8>(act_scales[2]),
                    ::aie::broadcast<float, 8>(act_scales[3]));
  ::aie::vector<int32, 8> b8 = ::aie::load_v<8>(bias);
  sums = ::aie::add(sums, ::aie::concat(b8, b8, b8, b8));
  ok &=
      (::aie::ge(sums, int32(-(1 << 23) + 1)) & ::aie::lt(sums, int32(1 << 23)))
          .count() == 32;
  ::aie::vector<float, 32> fp;
  if (ok)
    fp = vec_mul_rne(vec_mul_rne(::aie::to_float<float>(sums, 0), a_v), w_v);
  for (int m = 0; m < m_end; m++)
    for (int n = 0; n < 8; n++) {
      const int lane = m * 8 + n;
      float out_val;
      if (ok) {
        out_val = fp.get(lane);
      } else {
        // The target has no scalar float multiplier (only the int->float
        // cast above is unavoidable there): vec_mul_rne keeps this exact
        // without pulling in __mulsf3/__muldi3 for a path only the rare
        // overflow case reaches. 8 lanes: the narrowest AIE2P legalizes an
        // integer bit_and/bit_or at (see q_requant_heads).
        ::aie::vector<float, 8> sv =
            vec_mul_rne(::aie::broadcast<float, 8>((float)sums.get(lane)),
                        ::aie::broadcast<float, 8>(act_scales[m]));
        out_val =
            vec_mul_rne(sv, ::aie::broadcast<float, 8>(w_scales[n])).get(0);
      }
      reinterpret_cast<float *>(out)[(4 * h + m) * ostride + row0 + n] =
          out_val;
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
  half_epilogue(::aie::load_v<32>(sums), act_scales, w, out, 0, ostride, row0,
                m_count);
  if (m_count > 4)
    half_epilogue(::aie::load_v<32>(sums + 32), act_scales + 4, w, out, 1,
                  ostride, row0, m_count);
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

// llama_mma_f32 under a second symbol, which a design binds to a second
// output buffer type (one symbol takes one signature): a true linker alias,
// so the two symbols share one compiled body instead of a call (which would
// add a stack frame on top of llama_mma_f32's own) or a duplicate copy.
void llama_mma_f32_to(int8_t *restrict act, int8_t *restrict w,
                      int32_t *restrict part, int8_t *restrict out, int32_t g,
                      int32_t half, int32_t ostride, int32_t row0,
                      int32_t m_count) __attribute__((alias("llama_mma_f32")));

} // extern "C"

// The lm_head on the array. A table slot is 4 rows of the tied embed table,
// row-major (kLmRows x 2048), then 8 biases and 8 w_scales as a weight slot's
// trailer (rows 4-7: bias 0, scale 1). Each core keeps the top kLmKset of its
// share of the vocabulary by (logit, lower gidx), with their embed rows (in
// the idle mma tile) and their meta (in the idle partials), the lm_head core's
// candidate layout: per 4 entries 16 words, their logits, gidx and scales,
// then the set's length.
#ifndef LLAMA_LM_KSET
#define LLAMA_LM_KSET 8
#endif
static constexpr int kLmKset = LLAMA_LM_KSET;
static constexpr int kLmRows = 4;
static constexpr int kD = 2048;
static_assert(kLmKset == 8, "the set is the mma tile's 8 rows, 2 meta blocks");

static inline int32_t lm_key(float f) {
  int32_t b = __builtin_bit_cast(int32_t, f);
  if (b == INT32_MIN)
    b = 0;
  return b ^ ((b >> 31) & 0x7fffffff);
}

extern "C" {

// The logits of table slot `tile_idx` against act's first activation (a
// decode activation object: row 0's 2048 columns, its scale at 4 * 2048),
// inserted into the core's set. Tile 0 of the core's share resets the set.
void llama_mma_lm_topk(int8_t *restrict act, int8_t *restrict w,
                       int8_t *restrict rows, int32_t *restrict meta,
                       int32_t tile_idx, int32_t first) {
  event0();
  if (first)
    meta[12] = 0;

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
      *reinterpret_cast<const float *>(act + 4 * kD), 1.0f, 1.0f, 1.0f};
  alignas(32) float logits[8];
  const aie::saturation_mode saved_sat =
      ::aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  half_epilogue(sums, act_scales, w, reinterpret_cast<int8_t *>(logits), 0, 0,
                0, 1);
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);

  const float *w_scales = reinterpret_cast<const float *>(w + kTrailer + 32);
  int len = meta[12];
  int32_t min_key = 0, min_gidx = 0;
  int mi = 0;
  bool rescan = true;
  for (int n = 0; n < kLmRows; n++) {
    const int32_t gidx = tile_idx * kLmRows + n;
    const int32_t key = lm_key(logits[n]);
    int slot;
    if (len < kLmKset) {
      slot = len++;
    } else {
      if (rescan) {
        mi = 0;
        min_key = lm_key(__builtin_bit_cast(float, meta[0]));
        min_gidx = meta[4];
        for (int i = 1; i < kLmKset; i++) {
          const int at = 16 * (i / 4) + i % 4;
          int32_t k = lm_key(__builtin_bit_cast(float, meta[at]));
          if (k < min_key || (k == min_key && meta[at + 4] > min_gidx)) {
            min_key = k;
            min_gidx = meta[at + 4];
            mi = i;
          }
        }
        rescan = false;
      }
      if (key < min_key || (key == min_key && gidx > min_gidx))
        continue;
      slot = mi;
      rescan = true;
    }
    const int at = 16 * (slot / 4) + slot % 4;
    meta[at] = __builtin_bit_cast(int32_t, logits[n]);
    meta[at + 4] = gidx;
    meta[at + 8] = __builtin_bit_cast(int32_t, w_scales[n]);
    for (int c = 0; c < kD / 64; c++)
      ::aie::store_v(rows + slot * kD + c * 64,
                     ::aie::load_v<64>(w + n * kD + c * 64));
  }
  meta[12] = len;
  meta[28] = len;
  event1();
}

// Part j of the core's set for the lm_head core: j < 4 its rows 2j and
// 2j + 1, j = 4 its meta (the first 4096 bytes of the partials).
void llama_mma_lm_out(int8_t *restrict rows, int32_t *restrict meta,
                      int8_t *restrict out, int32_t j) {
  const int8_t *src =
      j < 4 ? rows + 2 * j * kD : reinterpret_cast<const int8_t *>(meta);
  for (int c = 0; c < 2 * kD / 64; c++)
    ::aie::store_v(out + c * 64, ::aie::load_v<64>(src + c * 64));
}

} // extern "C"

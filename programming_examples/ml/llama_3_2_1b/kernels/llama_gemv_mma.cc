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

// The f32 or (i8) gate epilogue on activations 4h..4h+3's 32 sums (lane
// (m - 4h)*8 + n), bias added, written to out[m*ostride + row0 + n] for the
// first m_count activations: the GEMV epilogue's arithmetic per element, on
// all 32 lanes at once where the GEMV's guards hold for every lane (see
// vec_epilogue_exact), else its scalar code per element.
__attribute__((noinline)) static void
half_epilogue(::aie::vector<int32, 32> sums, const int8_t *restrict act,
              const int8_t *restrict w, int8_t *restrict out, int h,
              int32_t ostride, int32_t row0, int32_t m_count, bool i8,
              float inv_out_scale) {
  const int32_t *bias = reinterpret_cast<const int32_t *>(w + kTrailer);
  const float *w_scales = reinterpret_cast<const float *>(w + kTrailer + 32);
  const float *act_scales =
      reinterpret_cast<const float *>(act + kScalesOff) + 4 * h;
  const int m_end = m_count - 4 * h < 4 ? m_count - 4 * h : 4;
  const uint32_t kLo = i8 ? 96 : 80, kHi = i8 ? 158 : 174;
  bool ok =
      !i8 ||
      (uint32_t)(((__builtin_bit_cast(int32_t, inv_out_scale) >> 23) & 0xff) -
                 kLo) <= kHi - kLo;
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
  if (ok) {
    fp = vec_mul_rne(vec_mul_rne(::aie::to_float<float>(sums, 0), a_v), w_v);
    if (i8) {
      fp = vec_mul_rne(fp, ::aie::broadcast<float, 32>(inv_out_scale));
      ok = ::aie::reduce_max(::aie::bit_and(
               int32(0x7fffffff), fp.template cast_to<int32>())) < 0x4f000000;
    }
  }
  ::aie::vector<int8, 32> r8;
  if (i8 && ok)
    r8 = vec_round_to_i8(fp);
  for (int m = 0; m < m_end; m++)
    for (int n = 0; n < 8; n++) {
      const int lane = m * 8 + n;
      const int at = (4 * h + m) * ostride + row0 + n;
      if (i8) {
        int8_t v;
        if (ok) {
          v = r8.get(lane);
        } else {
          float f = (float)sums.get(lane) * act_scales[m] * w_scales[n];
          float scaled = f * inv_out_scale;
          int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
          v = (int8_t)(r > I8_MAX ? I8_MAX : r < I8_MIN ? I8_MIN : r);
        }
        out[at] = v;
      } else {
        reinterpret_cast<float *>(out)[at] =
            ok ? fp.get(lane)
               : (float)sums.get(lane) * act_scales[m] * w_scales[n];
      }
    }
}

static inline void epilogue(const int8_t *restrict act,
                            const int8_t *restrict w,
                            const int32_t *restrict part, int8_t *restrict out,
                            int32_t half, int32_t ostride, int32_t row0,
                            int32_t m_count, bool i8, float inv_out_scale) {
  alignas(64) int32_t sums[64];
  half_sums(act, w, half, sums, part);
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  half_epilogue(::aie::load_v<32>(sums), act, w, out, 0, ostride, row0, m_count,
                i8, inv_out_scale);
  if (m_count > 4)
    half_epilogue(::aie::load_v<32>(sums + 32), act, w, out, 1, ostride, row0,
                  m_count, i8, inv_out_scale);
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
  epilogue(act, w, part + 64 * g, out, half, ostride, row0, m_count, false,
           0.0f);
  event1();
}

// llama_mma_f32 under a second symbol, which a design binds to a second
// output buffer type (one symbol takes one signature).
void llama_mma_f32_to(int8_t *restrict act, int8_t *restrict w,
                      int32_t *restrict part, int8_t *restrict out, int32_t g,
                      int32_t half, int32_t ostride, int32_t row0,
                      int32_t m_count) {
  llama_mma_f32(act, w, part, out, g, half, ostride, row0, m_count);
}

// As llama_mma_f32, through the int8 (gate) epilogue: out is int8.
void llama_mma_i8(int8_t *restrict act, int8_t *restrict w,
                  int32_t *restrict part, int8_t *restrict out, int32_t g,
                  int32_t half, int32_t ostride, int32_t row0, int32_t m_count,
                  float inv_out_scale) {
  event0();
  epilogue(act, w, part + 64 * g, out, half, ostride, row0, m_count, true,
           inv_out_scale);
  event1();
}

} // extern "C"

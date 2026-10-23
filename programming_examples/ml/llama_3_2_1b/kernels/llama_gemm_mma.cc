//===- llama_gemm_mma.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The prefill's GEMM cores on aie::mmul<8, 8, 8>: a C tile of 64 rows
// (activations) by 64 output columns (eight weight row groups), its K in
// chunks of 128. An A chunk is a 64 x 128 byte block x[r][b] tiled for mmul:
// x[r][b] at (r/8)*1024 + (b/8)*64 + (r%8)*8 + b%8; a W chunk is
// W[8g + n][8kt + i] at g*1024 + kt*64 + i*8 + n. C is int32 [8 row blocks]
// [8 groups][64], lane m*8 + n. The epilogue (llama_gemm_out) takes the 64
// rows' act_scales and the tile's 8 groups' weight slot trailers ([group][64
// bytes]), and is the GEMV's, so rows are byte-identical to the decode's.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"

static constexpr int kKt = 128 / 8;
static constexpr int kBlock = 8 * 128;
static constexpr int kChunk = 64 * 128;
static constexpr int kTrailerWords = 16;
static constexpr int kOutBlocks = 1;
using MMUL = ::aie::mmul<8, 8, 8, int8, int8, acc32>;

// llama_gemm_out's vector form, where its scale guards hold and the biases
// are zero: rows' 8 * kOutBlocks rows by 64 columns through epilogue_rne, a
// group's 4 rows (32 sums) at a time. Returns whether every sum was in
// (-2^23, 2^23), else out is to be rewritten. Out of line, so that its frame
// and half_epilogue's don't add up.
__attribute__((noinline)) static bool gemm_out_rne(const int32_t *restrict rows,
                                                   const float *restrict sc,
                                                   const int32_t *restrict tr,
                                                   float *restrict out) {
  constexpr int kSets = 2 * kOutBlocks;
  // epilogue_rne's operands: per 4 rows from their act_scales, per group
  // from its w_scales.
  alignas(64) int16_t row_lo[kSets][32], row_hi[kSets][32];
  alignas(64) int32_t row_e[kSets][32], row_e1[kSets][32];
  alignas(64) int16_t col_lo[8][32], col_hi[8][32];
  alignas(64) int32_t col_e[8][16];
  for (int q = 0; q < kSets; q++) {
    ::aie::vector<int32, 32> a_bits =
        ::aie::concat(::aie::broadcast<float, 8>(sc[4 * q]),
                      ::aie::broadcast<float, 8>(sc[4 * q + 1]),
                      ::aie::broadcast<float, 8>(sc[4 * q + 2]),
                      ::aie::broadcast<float, 8>(sc[4 * q + 3]))
            .template cast_to<int32>();
    EpilogueSig am = epilogue_sig(a_bits);
    ::aie::store_v(row_lo[q],
                   ::aie::vector<uint16, 32>(am.lo).template cast_to<int16>());
    ::aie::store_v(row_hi[q], ::aie::vector<int16, 32>(am.hi));
    ::aie::store_v(row_e[q], epilogue_exp(a_bits, 126));
    ::aie::store_v(row_e1[q], epilogue_exp(a_bits, 125));
  }
  for (int g = 0; g < 8; g++) {
    ::aie::vector<int32, 8> w8 = ::aie::load_v<8>(tr + g * kTrailerWords + 8);
    ::aie::vector<int32, 32> w_bits = ::aie::concat(w8, w8, w8, w8);
    EpilogueSig wm = epilogue_sig(w_bits);
    ::aie::store_v(col_lo[g],
                   ::aie::vector<uint16, 32>(wm.lo).template cast_to<int16>());
    ::aie::store_v(col_hi[g], ::aie::vector<int16, 32>(wm.hi));
    ::aie::store_v(col_e[g], epilogue_exp(w_bits, 127).template extract<16>(0));
  }
  uint32_t wide_s = 0;
  const int32_t *__restrict s_at = rows;
  // Item b (32 sums) is rows 4q..4q + 3 of group g. A trip takes item b's
  // first product and item b - 1's second, so that their chains overlap.
  constexpr int kItems = 16 * kOutBlocks;
  auto first = [&](int b) {
    const int q = (b >> 4) * 2 + (b & 1);
    return epilogue_first(
        ::aie::load_v<32>(s_at + 32 * b),
        {::aie::load_v<32>(row_lo[q]).template cast_to<uint16>(),
         ::aie::load_v<32>(row_hi[q])},
        wide_s);
  };
  auto rest = [&](const EpilogueState &x, int b) {
    const int q = (b >> 4) * 2 + (b & 1);
    const int g = (b >> 1) & 7;
    const ::aie::vector<int32, 16> ce = ::aie::load_v<16>(col_e[g]);
    ::aie::vector<float, 32> fp = epilogue_third(epilogue_second(
        x,
        {::aie::load_v<32>(col_lo[g]).template cast_to<uint16>(),
         ::aie::load_v<32>(col_hi[g])},
        ::aie::load_v<32>(row_e[q]), ::aie::load_v<32>(row_e1[q]),
        ::aie::concat(ce, ce)));
    float *o = out + 4 * q * 64 + g * 8;
#pragma clang loop unroll(full)
    for (int r = 0; r < 4; r++)
      ::aie::store_v(o + r * 64, fp.template extract<8>(r));
  };
  EpilogueState x = first(0);
  for (int b = 1; b < kItems; b++) {
    EpilogueState next = first(b);
    rest(x, b - 1);
    x = next;
  }
  rest(x, kItems - 1);
  return wide_s == 0;
}

extern "C" {

// C (first: =, else +=) the chunk's a x W.
void llama_gemm_chunk(const int8_t *restrict a, const int8_t *restrict w,
                      int32_t *restrict c, int32_t first) {
  event0();
  for (int mb = 0; mb < 8; mb += 2)
    for (int g = 0; g < 8; g += 2) {
      int32_t *c00 = c + (mb * 8 + g) * 64;
      int32_t *c01 = c00 + 64;
      int32_t *c10 = c00 + 8 * 64;
      int32_t *c11 = c10 + 64;
      const int8_t *__restrict a0 = a + mb * kBlock;
      const int8_t *__restrict a1 = a0 + kBlock;
      const int8_t *__restrict w0 = w + g * 1024;
      const int8_t *__restrict w1 = w0 + 1024;
      ::aie::vector<int8, 64> va0 = ::aie::load_v<64>(a0);
      ::aie::vector<int8, 64> va1 = ::aie::load_v<64>(a1);
      ::aie::vector<int8, 64> vw0 = ::aie::load_v<64>(w0);
      ::aie::vector<int8, 64> vw1 = ::aie::load_v<64>(w1);
      MMUL m00, m01, m10, m11;
      if (first) {
        m00.mul(va0, vw0);
        m01.mul(va0, vw1);
        m10.mul(va1, vw0);
        m11.mul(va1, vw1);
      } else {
        m00 = MMUL(::aie::accum<acc32, 64>(::aie::load_v<64>(c00)));
        m01 = MMUL(::aie::accum<acc32, 64>(::aie::load_v<64>(c01)));
        m10 = MMUL(::aie::accum<acc32, 64>(::aie::load_v<64>(c10)));
        m11 = MMUL(::aie::accum<acc32, 64>(::aie::load_v<64>(c11)));
        m00.mac(va0, vw0);
        m01.mac(va0, vw1);
        m10.mac(va1, vw0);
        m11.mac(va1, vw1);
      }
      for (int kt = 1; kt < kKt; kt++) {
        a0 += 64;
        a1 += 64;
        w0 += 64;
        w1 += 64;
        va0 = ::aie::load_v<64>(a0);
        va1 = ::aie::load_v<64>(a1);
        vw0 = ::aie::load_v<64>(w0);
        vw1 = ::aie::load_v<64>(w1);
        m00.mac(va0, vw0);
        m01.mac(va0, vw1);
        m10.mac(va1, vw0);
        m11.mac(va1, vw1);
      }
      ::aie::store_v(c00, m00.to_accum().template to_vector<int32>());
      ::aie::store_v(c01, m01.to_accum().template to_vector<int32>());
      ::aie::store_v(c10, m10.to_accum().template to_vector<int32>());
      ::aie::store_v(c11, m11.to_accum().template to_vector<int32>());
    }
  event1();
}

// A projection's trailers, [core][tile][group][64 bytes] in kChunk-byte A
// chunks: chunk i's part of core `core`'s (of `tiles` tiles) into trailers.
void llama_gemm_take_trailers(const int8_t *restrict chunk,
                              int32_t *restrict trailers, int32_t i,
                              int32_t core, int32_t tiles) {
  const int32_t len = tiles * 8 * kTrailerWords * 4;
  const int32_t lo = core * len > i * kChunk ? core * len : i * kChunk;
  const int32_t hi =
      (core + 1) * len < (i + 1) * kChunk ? (core + 1) * len : (i + 1) * kChunk;
  for (int32_t b = lo; b < hi; b += 64)
    ::aie::store_v(reinterpret_cast<int8_t *>(trailers) + b - core * len,
                   ::aie::load_v<64>(chunk + b - i * kChunk));
}

// The 64 rows' act_scales, row r's at the start of its row of the chunk.
void llama_gemm_act_scales(const int8_t *restrict chunk,
                           float *restrict act_scales) {
  for (int r = 0; r < 64; r++)
    act_scales[r] = *reinterpret_cast<const float *>(chunk + (r >> 3) * kBlock +
                                                     (r & 7) * 8);
}

// Part `part` of the C tile, its rows 8mb..8mb + 8 * kOutBlocks - 1 for
// mb = part * kOutBlocks, through the GEMV's epilogue, to out (fp32
// [rows][64]): gemm_out_rne where every scale is positive with its exponent
// field in [kEpilogueExpLo, kEpilogueExpHi], every bias is zero (as the
// model's are) and every sum is in (-2^23, 2^23); otherwise per 4 rows of a
// group, as half_epilogue decides.
void llama_gemm_out(const int32_t *restrict c, const float *restrict act_scales,
                    const int32_t *restrict trailers, float *restrict out,
                    int32_t part, int32_t core) {
  event0();
  const int32_t mb = part * kOutBlocks;
  const aie::saturation_mode saved_sat =
      ::aie::swap_saturation(aie::saturation_mode::saturate);
  const aie::rounding_mode saved_rnd =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  const float *sc = act_scales + mb * 8;
  const int32_t *tr = trailers + core * 8 * kTrailerWords;
  const int32_t *rows = c + mb * 8 * 64;
  // The part's act_scales, then the 8 groups' w_scales, 16 at a time; a
  // negative scale's sign and exponent bits compare below the range as int32.
  ::aie::mask<16> ok = ::aie::mask<16>(true);
  for (int i = 0; i < kOutBlocks + 4; i++) {
    const int32_t *lo = i < kOutBlocks
                            ? reinterpret_cast<const int32_t *>(sc) + 8 * i
                            : tr + 2 * (i - kOutBlocks) * kTrailerWords + 8;
    const int32_t *hi = i < kOutBlocks ? lo : lo + kTrailerWords;
    ::aie::vector<int32, 16> e =
        ::aie::bit_and(int32(0xff800000), ::aie::concat(::aie::load_v<8>(lo),
                                                        ::aie::load_v<8>(hi)));
    ok = ok & ::aie::ge(e, int32(kEpilogueExpLo << 23)) &
         ::aie::le(e, int32(kEpilogueExpHi << 23));
  }
  for (int g = 0; g < 8; g += 2)
    ok = ok & ::aie::eq(
                  ::aie::concat(::aie::load_v<8>(tr + g * kTrailerWords),
                                ::aie::load_v<8>(tr + (g + 1) * kTrailerWords)),
                  int32(0));
  if (!ok.full() || !gemm_out_rne(rows, sc, tr, out))
    for (int g = 0; g < 8; g++)
      for (int i = 0; i < 2 * kOutBlocks; i++)
        half_epilogue(
            ::aie::load_v<32>(rows + (i / 2 * 8 + g) * 64 + 32 * (i % 2)),
            sc + 4 * i,
            reinterpret_cast<const int8_t *>(tr + g * kTrailerWords),
            reinterpret_cast<int8_t *>(out + i / 2 * 8 * 64), i % 2, 64, g * 8,
            8);
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}

} // extern "C"

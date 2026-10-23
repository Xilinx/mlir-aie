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
// [8 groups][64], lane m*8 + n. The epilogue (on another core) takes the 64
// rows' act_scales with each projection and the 8 cores' 8 groups' weight
// slot trailers ([core][group][64 bytes]) with each tile. The epilogue is
// the GEMV's, so rows are byte-identical to the decode's.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "llama_gemv_array.h"

static constexpr int kKt = 128 / 8;
static constexpr int kBlock = 8 * 128;
static constexpr int kTrailerWords = 16;
static constexpr int kOutBlocks = 1;
using MMUL = ::aie::mmul<8, 8, 8, int8, int8, acc32>;

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

// Part `part` of the C tile, its rows 8mb..8mb + 8 * kOutBlocks - 1 for
// mb = part * kOutBlocks, through the GEMV's epilogue, to out (fp32
// [rows][64]): its vector form, stored a vector at a time, with the guards
// checked for all the rows at once; where they fail, per 4 rows of a group
// as half_epilogue decides, which takes the rows whose guards fail.
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
  // The guards of half_epilogue_vec for every row group at once: each scale's
  // exponent field in [kEpilogueExpLo, kEpilogueExpHi] (the masked bits
  // compare as nonnegative int32), every sum plus bias in (-2^23, 2^23).
  const ::aie::vector<int32, 16> exp_lo =
      ::aie::broadcast<int32, 16>(int32(kEpilogueExpLo << 23));
  const ::aie::vector<int32, 16> exp_hi =
      ::aie::broadcast<int32, 16>(int32(kEpilogueExpHi << 23));
  ::aie::mask<16> scales_ok = ::aie::mask<16>(true);
  for (int i = 0; i < 8 * kOutBlocks + 64; i += 16) {
    ::aie::vector<int32, 16> e = ::aie::bit_and(
        int32(0x7f800000),
        i < 8 * kOutBlocks
            ? ::aie::load_v<16>(reinterpret_cast<const int32_t *>(sc) + i)
            : ::aie::concat(
                  ::aie::load_v<8>(
                      tr + (i - 8 * kOutBlocks) / 8 * kTrailerWords + 8),
                  ::aie::load_v<8>(
                      tr + ((i - 8 * kOutBlocks) / 8 + 1) * kTrailerWords +
                      8)));
    scales_ok = scales_ok & ::aie::ge(e, exp_lo) & ::aie::le(e, exp_hi);
  }
  ::aie::vector<int32, 32> s_min =
      ::aie::broadcast<int32, 32>(int32((1 << 23) - 1));
  ::aie::vector<int32, 32> s_max =
      ::aie::broadcast<int32, 32>(int32(-(1 << 23) + 1));
  for (int g = 0; g < 8; g++) {
    ::aie::vector<int32, 8> b8 = ::aie::load_v<8>(tr + g * kTrailerWords);
    ::aie::vector<int32, 32> b_v = ::aie::concat(b8, b8, b8, b8);
    for (int i = 0; i < 2 * kOutBlocks; i++) {
      ::aie::vector<int32, 32> s = ::aie::add(
          ::aie::load_v<32>(rows + (i / 2 * 8 + g) * 64 + 32 * (i % 2)), b_v);
      s_min = ::aie::min(s_min, s);
      s_max = ::aie::max(s_max, s);
    }
  }
  if (scales_ok.full() && ::aie::reduce_min(s_min) > -(1 << 23) &&
      ::aie::reduce_max(s_max) < (1 << 23)) {
    // vec_mul_rne's split of each product's shared operand, the rows'
    // act_scales and each group's w_scales (exponent fields masked before
    // the shift, which may round).
    ::aie::vector<int32, 32> a_bits[2 * kOutBlocks], a_e[2 * kOutBlocks],
        a_m[2 * kOutBlocks];
    for (int i = 0; i < 2 * kOutBlocks; i++) {
      a_bits[i] = ::aie::concat(::aie::broadcast<float, 8>(sc[4 * i]),
                                ::aie::broadcast<float, 8>(sc[4 * i + 1]),
                                ::aie::broadcast<float, 8>(sc[4 * i + 2]),
                                ::aie::broadcast<float, 8>(sc[4 * i + 3]))
                      .template cast_to<int32>();
      a_e[i] =
          ::aie::downshift(::aie::bit_and(int32(0x7f800000), a_bits[i]), 23);
      a_m[i] = ::aie::bit_or(int32(0x800000),
                             ::aie::bit_and(int32(0x7fffff), a_bits[i]));
    }
    for (int g = 0; g < 8; g++) {
      ::aie::vector<int32, 8> b8 = ::aie::load_v<8>(tr + g * kTrailerWords);
      ::aie::vector<int32, 8> w8 = ::aie::load_v<8>(tr + g * kTrailerWords + 8);
      ::aie::vector<int32, 32> b_v = ::aie::concat(b8, b8, b8, b8);
      ::aie::vector<int32, 32> w_bits = ::aie::concat(w8, w8, w8, w8);
      ::aie::vector<int32, 32> w_e =
          ::aie::downshift(::aie::bit_and(int32(0x7f800000), w_bits), 23);
      ::aie::vector<int32, 32> w_m = ::aie::bit_or(
          int32(0x800000), ::aie::bit_and(int32(0x7fffff), w_bits));
      for (int i = 0; i < 2 * kOutBlocks; i++) {
        ::aie::vector<int32, 32> s = ::aie::add(
            ::aie::load_v<32>(rows + (i / 2 * 8 + g) * 64 + 32 * (i % 2)), b_v);
        ::aie::vector<float, 32> fp =
            vec_mul_rne_split(vec_mul_rne_split(::aie::to_float<float>(s, 0),
                                                a_bits[i], a_e[i], a_m[i]),
                              w_bits, w_e, w_m);
        for (int m = 0; m < 4; m++)
          ::aie::store_v(out + (4 * i + m) * 64 + g * 8, fp.extract<8>(m));
      }
    }
  } else {
    for (int g = 0; g < 8; g++)
      for (int i = 0; i < 2 * kOutBlocks; i++) {
        const int8_t *trailer =
            reinterpret_cast<const int8_t *>(tr + g * kTrailerWords);
        ::aie::vector<int32, 32> sums =
            ::aie::load_v<32>(rows + (i / 2 * 8 + g) * 64 + 32 * (i % 2));
        bool half_ok;
        ::aie::vector<float, 32> fp =
            half_epilogue_vec(sums, sc + 4 * i, trailer, half_ok);
        if (half_ok)
          for (int m = 0; m < 4; m++)
            ::aie::store_v(out + (4 * i + m) * 64 + g * 8, fp.extract<8>(m));
        else
          half_epilogue(
              ::aie::load_v<32>(rows + (i / 2 * 8 + g) * 64 + 32 * (i % 2)),
              sc + 4 * i, trailer,
              reinterpret_cast<int8_t *>(out + i / 2 * 8 * 64), i % 2, 64,
              g * 8, 8);
      }
  }
  aie::set_rounding(saved_rnd);
  aie::set_saturation(saved_sat);
  event1();
}

} // extern "C"

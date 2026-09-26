//===- rgba2hue.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

// clang-format off
#include <aie_api/aie.hpp>
#include "../aie_kernel_utils.h"
#include "lut_inv.h"
// clang-format on

const int32_t SRS_SHIFT = 12;

__attribute__((inline)) void xf_extract_rgb(uint8_t *ptr_rgba,
                                            ::aie::vector<uint8_t, 32> &r,
                                            ::aie::vector<uint8_t, 32> &g,
                                            ::aie::vector<uint8_t, 32> &b) {
  ::aie::vector<uint8_t, 32> rgba_channel0, rgba_channel1, rgba_channel3,
      rgba_channel2;
  rgba_channel0 = ::aie::load_v<32>(ptr_rgba);
  ptr_rgba += 32;
  rgba_channel1 = ::aie::load_v<32>(ptr_rgba);
  ptr_rgba += 32;
  rgba_channel2 = ::aie::load_v<32>(ptr_rgba);
  ptr_rgba += 32;
  rgba_channel3 = ::aie::load_v<32>(ptr_rgba);
  ptr_rgba += 32;

  // Unzip the interleaved channels
  auto [rg_temp, ba_temp] =
      ::aie::interleave_unzip(::aie::concat(rgba_channel0, rgba_channel1),
                              ::aie::concat(rgba_channel2, rgba_channel3), 2);
  r = ::aie::filter_even(rg_temp, 1);
  g = ::aie::filter_odd(rg_temp, 1);
  b = ::aie::filter_even(ba_temp, 1);
}

__attribute__((inline)) void
comp_divisor_16b(::aie::vector<uint8_t, 32> divisor,
                 ::aie::vector<uint16_t, 32> &divisor_select) {
  const int step = 0;
  using lut_type_uint16 = aie::lut<4, uint16, uint16>;
  lut_type_uint16 inv_lut_16b(num_entries_lut_inv_16b, lut_inv_16b_ab.data(),
                              lut_inv_16b_cd.data());
  aie::parallel_lookup<uint8, lut_type_uint16, aie::lut_oor_policy::truncate>
      lookup_inv_16b(inv_lut_16b, step);

  aie::vector<uint8, 16> input1, input2;
  aie::vector<uint16, 16> res1, res2;
  input1 = divisor.extract<16>(0);
  input2 = divisor.extract<16>(1);
  res1 = lookup_inv_16b.fetch(input1.cast_to<uint8>());
  res2 = lookup_inv_16b.fetch(input2.cast_to<uint8>());
  divisor_select = aie::concat(res1, res2);
}

#if AIE_TUNED_AIE2P
// comp_divisor_16b's lookups without aie::parallel_lookup, whose indices go
// through a 64-bit accumulator: switching the accumulator mode between them
// and the hue's 32-bit macs serialized every iteration. The indices here are
// the divisor times the four bytes of a doubled entry, in 32 bits.
__attribute__((inline)) ::aie::vector<uint16_t, 32>
lookup_inv_16b(::aie::vector<uint8_t, 32> divisor) {
  ::aie::accum<acc32, 32> index;
  index.from_vector(divisor, 2);
  v64int8 c0, c1, c2, c3;
  ::load_lut_2x_int8((int *)lut_inv_16b_ab.data(), (int *)lut_inv_16b_cd.data(),
                     index.extract<16>(0).to_vector<int32>(0), c0, c1);
  ::load_lut_2x_int8((int *)lut_inv_16b_ab.data(), (int *)lut_inv_16b_cd.data(),
                     index.extract<16>(1).to_vector<int32>(0), c2, c3);
  c0 = ::shuffle(c0, c1, T16_16x4_lo);
  c2 = ::shuffle(c2, c3, T16_16x4_lo);
  return ::aie::vector<uint16_t, 32>((v32uint16)::shuffle(c0, c2, T256_2x2_lo));
}
#endif

#if AIE_TUNED_AIE2P
// The max channel picks the one hue formula a pixel needs, so select its
// operands and base first and run one mac pair instead of all three. The
// precedence is the same, g over r over b, and falls out of the comparisons
// the max takes: r wins where g < r, b where it beats both strictly. A zero
// divisor takes r's formula, whose base alone rounds to 0. The base is a mac
// of 0, 2 or 4 by 170 << 8, for 0, 170 or 340, and the rounding mode adds the
// 1 (half of the final shift) that the other kernels put in the accumulator.
template <unsigned N>
__attribute__((always_inline)) ::aie::vector<uint8_t, N>
hue_of(::aie::vector<uint8_t, N> r, ::aie::vector<uint8_t, N> g,
       ::aie::vector<uint8_t, N> b) {
  auto [rg, r_wins] = ::aie::max_cmp(g, r);
  auto [rgbMax, b_wins] = ::aie::max_cmp(rg, b);
  auto divisor = ::aie::sub(rgbMax, ::aie::min(::aie::min(r, g), b));
  ::aie::vector<uint16_t, N> divisor_sel;
  if constexpr (N == 32)
    divisor_sel = lookup_inv_16b(divisor);
  else
    divisor_sel =
        ::aie::concat(lookup_inv_16b(divisor.template extract<32>(0)),
                      lookup_inv_16b(divisor.template extract<32>(1)));
  ::aie::vector<uint8_t, N> zero = ::aie::zeros<uint8_t, N>();
  r_wins = r_wins | ::aie::eq(divisor, zero);
  auto p = ::aie::select(::aie::select(b, g, r_wins), r, b_wins);
  auto q = ::aie::select(::aie::select(r, b, r_wins), g, b_wins);
  auto k = ::aie::select(
      ::aie::select(::aie::broadcast<uint8_t, N>(2), zero, r_wins),
      ::aie::broadcast<uint8_t, N>(4), b_wins);
  ::aie::accum<acc32, N> h = ::aie::mul(p, divisor_sel);
  h = ::aie::msc(h, q, divisor_sel);
  h = ::aie::mac(h, k, ::aie::broadcast<uint16_t, N>(170 << 8));
  return h.template to_vector<uint8_t>(10);
}
#endif

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
// As in rgba2gray.cc (see there), the loop pipelines only when known to run
// at least four times, and here also only as the one copy of the body in its
// function; a shorter row takes the plain loop in its own function.
template <bool MinFour>
__attribute__((noinline)) void
rgba2hue_rows(uint8_t *__restrict rgba_in, uint8_t *__restrict hue_out,
#else
__attribute__((noinline)) void
rgba2hue_aie(uint8_t *rgba_in, uint8_t *hue_out,
#endif
              const int32_t height, const int32_t width) {
  event0();
#if AIE_TUNED_AIE2P
  const ::aie::rounding_mode rounding = ::aie::get_rounding();
  ::aie::set_rounding(::aie::rounding_mode::positive_inf);
#endif
  ::aie::vector<uint8_t, 32> r, g, b;
  ::aie::vector<uint8_t, 32> hue;

  ::aie::vector<uint8_t, 32> rgbMin, rgbMax;

  ::aie::vector<uint8_t, 32> zero32 = aie::zeros<uint8_t, 32>();

  ::aie::vector<int16_t, 32> eightFive = aie::zeros<int16_t, 32>();
  eightFive[0] = 85;
  eightFive[1] = -85;
  ::aie::vector<int16_t, 32> one = aie::broadcast<int16_t, 32>(1);
  ::aie::vector<int16_t, 32> twoEightFive =
      aie::broadcast<int16_t, 32>(171); // 170 + 1
  ::aie::vector<int16_t, 32> fourEightFive =
      aie::broadcast<int16_t, 32>(341); // 340 + 1

#if AIE_TUNED_AIE2P
  // 64 pixels a step, split into channels as in rgba2gray.cc.
  auto body = [&]() __attribute__((always_inline)) {
    ::aie::vector<uint8_t, 64> v0 = ::aie::load_v<64>(rgba_in);
    ::aie::vector<uint8_t, 64> v1 = ::aie::load_v<64>(rgba_in + 64);
    ::aie::vector<uint8_t, 64> v2 = ::aie::load_v<64>(rgba_in + 128);
    ::aie::vector<uint8_t, 64> v3 = ::aie::load_v<64>(rgba_in + 192);
    auto [rg01, ba01] = ::aie::interleave_unzip(v0, v1, 2);
    auto [rg23, ba23] = ::aie::interleave_unzip(v2, v3, 2);
    auto [r64, g64] = ::aie::interleave_unzip(rg01, rg23, 1);
    auto b64 = ::aie::interleave_unzip(ba01, ba23, 1).first;
    ::aie::store_v(hue_out, hue_of<64>(r64, g64, b64));
    rgba_in += 256;
    hue_out += 64;
  };
#else
  auto body = [&]() __attribute__((always_inline)) {
    xf_extract_rgb(rgba_in, r, g, b);

    // Get rgbMin and rgbMax
    rgbMin = ::aie::min(::aie::min(r, g), b);
    rgbMax = ::aie::max(::aie::max(r, g), b);

    // Get divisor and select the fixed point divisor to multiply by
    auto divisor = ::aie::sub(rgbMax, rgbMin);
    ::aie::vector<uint16, 32> divisor_sel;
    comp_divisor_16b(divisor, divisor_sel);

    // Initialize accum with value since 340 is larger than uint8
    aie::accum<acc32, 32> hr_partial(one, 9);
    aie::accum<acc32, 32> hg_partial(twoEightFive, 9);
    aie::accum<acc32, 32> hb_partial(fourEightFive, 9);

    // Performa uin8*int16 vector multiply
    hr_partial = aie::mac(hr_partial, g, divisor_sel);
    hg_partial = aie::mac(hg_partial, b, divisor_sel);
    hb_partial = aie::mac(hb_partial, r, divisor_sel);

    hr_partial = aie::msc(hr_partial, b, divisor_sel);
    hg_partial = aie::msc(hg_partial, r, divisor_sel);
    hb_partial = aie::msc(hb_partial, g, divisor_sel);

    auto hr = hr_partial.to_vector<uint8>(10); // Q7.9 shift + 1 (div 2)
    auto hg = hg_partial.to_vector<uint8>(10); // Q7.9 shift + 1 (div 2)
    auto hb = hb_partial.to_vector<uint8>(10); // Q7.9 shift + 1 (div 2)

    aie::mask<32> sel1 = aie::eq(rgbMax, r);
    auto tmp1 = aie::select(hb, hr, sel1);
    aie::mask<32> sel2 = aie::eq(rgbMax, g);
    auto tmp2 = aie::select(tmp1, hg, sel2);
    aie::mask<32> sel3 = aie::eq(divisor, zero32);
    hue = aie::select(tmp2, zero32, sel3);

    ::aie::store_v(hue_out, hue);
    rgba_in += 128;
    hue_out += 32;
  };
#endif
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
#if AIE_TUNED_AIE2P
  const int steps = (width * height) / 64;
#else
  const int steps = (width * height) / 32;
#endif
  if constexpr (MinFour) {
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int j = 0; j < steps; j++)
      body();
  } else {
    AIE_LOOP_NO_UNROLL
    for (int j = 0; j < steps; j++)
      body();
  }
#if AIE_TUNED_AIE2P
  if ((width * height) % 64) {
    xf_extract_rgb(rgba_in, r, g, b);
    ::aie::store_v(hue_out, hue_of<32>(r, g, b));
  }
  ::aie::set_rounding(rounding);
#endif
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int j = 0; (j < (width * height) / 32); j += 1) {
    body();
  }
#endif
  event1();
}

#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
void rgba2hue_aie(uint8_t *rgba_in, uint8_t *hue_out, const int32_t height,
                  const int32_t width) {
#if AIE_TUNED_AIE2P
  if ((width * height) / 64 >= 4)
#else
  if ((width * height) / 32 >= 4)
#endif
    rgba2hue_rows<true>(rgba_in, hue_out, height, width);
  else
    rgba2hue_rows<false>(rgba_in, hue_out, height, width);
}
#endif

extern "C" {

void rgba2hueLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
  rgba2hue_aie(in, out, 1, lineWidth);
}

void rgba2hueTile(uint8_t *in, uint8_t *out, int32_t tileHeight,
                  int32_t tileWidth) {
  rgba2hue_aie(in, out, tileHeight, tileWidth);
}

} // extern "C"

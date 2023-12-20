//===- rgba2hue.cc ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
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
#include "lut_inv_8b.h"
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
  lut_type_uint16 inv_lut_16b(num_entries_lut_inv_16b, lut_inv_16b_ab,
                              lut_inv_16b_cd);
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

__attribute__((noinline)) void rgba2hue_aie(uint8_t *rgba_in, uint8_t *hue_out,
                                            const int32_t height,
                                            const int32_t width) {
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

  for (int j = 0; (j < (width * height) / 32); j += 1)
    chess_prepare_for_pipelining {
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
    }
}

void rgba2hue_aie_scalar(uint8_t *rgba_in, uint8_t *hue_out,
                         const int32_t height, const int32_t width) {
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      int r = (int)rgba_in[i * (width * 4) + (j * 4)];
      int g = (int)rgba_in[i * (width * 4) + (j * 4) + 1];
      int b = (int)rgba_in[i * (width * 4) + (j * 4) + 2];
      int h;
      uint8_t rgbMin, rgbMax;

      rgbMin = r < g ? (r < b ? r : b) : (g < b ? g : b);
      rgbMax = r > g ? (r > b ? r : b) : (g > b ? g : b);

      if (rgbMax == 0 || rgbMax == rgbMin)
        h = 0;
      else if (rgbMax == r)
        h = 0 +
            85 * (g - b) /
                (rgbMax - rgbMin); // h = 0 + 42.5*(g - b) / (rgbMax - rgbMin);
      else if (rgbMax == g)
        h = 85 * 2 +
            85 * (b - r) /
                (rgbMax - rgbMin); // h = 85 + 42.5*(b - r) / (rgbMax - rgbMin);
      else
        h = 170 * 2 +
            85 * (r - g) /
                (rgbMax -
                 rgbMin); // h = 170 + 42.5*(r - g) / (rgbMax - rgbMin);

      h = (h + 1) >> 1;
      hue_out[i * width + j] = (uint8_t)h;
    }

  return;
}

extern "C" {

void rgba2hueLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
  // rgba2hue_aie_scalar(in, out, 1, lineWidth);
  rgba2hue_aie(in, out, 1, lineWidth);
}

void rgba2hueTile(uint8_t *in, uint8_t *out, int32_t tileHeight,
                  int32_t tileWidth) {
  rgba2hue_aie_scalar(in, out, tileHeight, tileWidth);
}

} // extern "C"

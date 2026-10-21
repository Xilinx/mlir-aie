//===- rgba2gray.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

const int32_t SRS_SHIFT = 15;
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

__attribute__((noinline)) void rgba2gray_aie(uint8_t *__restrict rgba_in,
                                             uint8_t *__restrict y_out,
                                             const int32_t height,
                                             const int32_t width) {
  event0();
  //::aie::vector<int16_t, 16> WT(66, 129, 25, 128); //Y=0.299*R + 0.587*G +
  //: 0.114*B (BT.470) :aie::vector<int16_t, 16> WT(25, 129, 66, 128);
  //://Y=0.299*R + 0.587*G + 0.114*B (BT.470)
  ::aie::vector<int16_t, 16> WT(
      (int16_t)round(0.299 * (1 << SRS_SHIFT)),
      (int16_t)round(0.587 * (1 << SRS_SHIFT)),
      (int16_t)round(0.114 * (1 << SRS_SHIFT)),
      (1 << (SRS_SHIFT - 1))); // Y=0.299*R + 0.587*G + 0.114*B (BT.470)
  ::aie::vector<uint8_t, 32> c1 = ::aie::broadcast<uint8_t, 32>(1);
  ::aie::vector<uint8_t, 32> r, g, b;
  ::aie::vector<uint8_t, 32> y;

#if AIE_TUNED_AIE2
  // The rounding term seeds the accumulator, leaving a chain of three macs
  // instead of a mul and three macs. Kept rolled, the loop then pipelines when
  // it is known to run at least six times; the count drops the loop's
  // zero-trip guard, so a shorter row takes the plain loop.
  ::aie::accum<acc32, 32> rnd;
  rnd.from_vector(::aie::broadcast<int32_t, 32>(1 << (SRS_SHIFT - 1)));
  auto body = [&]() __attribute__((always_inline)) {
    xf_extract_rgb(rgba_in, r, g, b);
    ::aie::accum<acc32, 32> acc =
        ::aie::mac(::aie::mac(::aie::mac(rnd, r, WT[0]), g, WT[1]), b, WT[2]);
    y = acc.template to_vector<uint8_t>(SRS_SHIFT);
    ::aie::store_v(y_out, y);
    rgba_in += 128;
    y_out += 32;
  };
  if ((width * height) / 32 >= 6) {
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(6)
    for (int j = 0; (j < (width * height) / 32); j += 1) {
      body();
    }
  } else {
    AIE_LOOP_NO_UNROLL
    for (int j = 0; (j < (width * height) / 32); j += 1) {
      body();
    }
  }
#elif AIE_TUNED_AIE2P
  // AIE2's rounding-seeded chain at AIE2P's width: 64 pixels a step from four
  // 512-bit loads, split into channels by two rounds of unzip.
  ::aie::accum<acc32, 64> rnd64;
  rnd64.from_vector(::aie::broadcast<int32_t, 64>(1 << (SRS_SHIFT - 1)));
  auto body64 = [&]() __attribute__((always_inline)) {
    ::aie::vector<uint8_t, 64> v0 = ::aie::load_v<64>(rgba_in);
    ::aie::vector<uint8_t, 64> v1 = ::aie::load_v<64>(rgba_in + 64);
    ::aie::vector<uint8_t, 64> v2 = ::aie::load_v<64>(rgba_in + 128);
    ::aie::vector<uint8_t, 64> v3 = ::aie::load_v<64>(rgba_in + 192);
    auto [rg01, ba01] = ::aie::interleave_unzip(v0, v1, 2);
    auto [rg23, ba23] = ::aie::interleave_unzip(v2, v3, 2);
    auto [r64, g64] = ::aie::interleave_unzip(rg01, rg23, 1);
    auto b64 = ::aie::interleave_unzip(ba01, ba23, 1).first;
    ::aie::accum<acc32, 64> acc = ::aie::mac(
        ::aie::mac(::aie::mac(rnd64, r64, WT[0]), g64, WT[1]), b64, WT[2]);
    ::aie::store_v(y_out, acc.template to_vector<uint8_t>(SRS_SHIFT));
    rgba_in += 256;
    y_out += 64;
  };
  const int steps = (width * height) / 64;
  if (steps >= 4) {
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int j = 0; j < steps; j++)
      body64();
  } else {
    AIE_LOOP_NO_UNROLL
    for (int j = 0; j < steps; j++)
      body64();
  }
  if ((width * height) % 64) {
    xf_extract_rgb(rgba_in, r, g, b);
    ::aie::accum<acc32, 32> acc = ::aie::mac(
        ::aie::mac(::aie::mac(rnd64.template extract<32>(0), r, WT[0]), g,
                   WT[1]),
        b, WT[2]);
    ::aie::store_v(y_out, acc.template to_vector<uint8_t>(SRS_SHIFT));
  }
#else
  AIE_PREPARE_FOR_PIPELINING
  for (int j = 0; (j < (width * height) / 32); j += 1) {
    xf_extract_rgb(rgba_in, r, g, b);

    ::aie::accum<acc32, 32> acc;
    acc = ::aie::accumulate<32>(WT, 0, r, g, b, c1);
    y = acc.template to_vector<uint8_t>(SRS_SHIFT);

    ::aie::store_v(y_out, y);
    rgba_in += 128;
    y_out += 32;
  }
#endif
  event1();
}

void rgba2gray_aie_scalar(uint8_t *rgba_in, uint8_t *y_out,
                          const int32_t height, const int32_t width) {
  /// Y=0.299*R + 0.587*G + 0.114*B (BT.470)
  const int colorMatrix[4] = {(int)round(0.299 * 65536),
                              (int)round(0.587 * 65536),
                              (int)round(0.114 * 65536), (65536 / 2)};
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      int r = (int)rgba_in[i * width * 4 + j * 4];
      int g = (int)rgba_in[i * width * 4 + j * 4 + 1];
      int b = (int)rgba_in[i * width * 4 + j * 4 + 2];
      int tmpSum = (colorMatrix[0] * r + colorMatrix[1] * g +
                    colorMatrix[2] * b + colorMatrix[3]) >>
                   16;
      y_out[i * width + j] = (uint8_t)tmpSum;
    }

  return;
}

extern "C" {

void rgba2grayLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
  rgba2gray_aie(in, out, 1, lineWidth);
}

void rgba2grayTile(uint8_t *in, uint8_t *out, int32_t tileHeight,
                   int32_t tileWidth) {
  rgba2gray_aie(in, out, tileHeight, tileWidth);
}

} // extern "C"

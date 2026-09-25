//===- gray2rgba.cc -------------------------------------------*- C++ -*-===//
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

::aie::vector<uint8, 64> vector_broadcast(::aie::vector<uint8, 16> e) {
  v64uint8 lli = e.template grow<64>();
  lli = shuffle(lli, lli, T8_2x64_lo);
  lli = shuffle(lli, lli, T8_2x64_lo);
  return ::aie::vector<uint8, 64>(lli);
}

void gray2rgba_aie(uint8_t *AIE2_RESTRICT y_in, uint8_t *AIE2_RESTRICT rgba_out,
                   const int32_t height, const int32_t width) {
  event0();
  // Initialize alpha vector
  ::aie::vector<uint8, 64> alpha255 = ::aie::zeros<uint8, 64>();
  for (int i = 0; i < 16; i++) {
    alpha255[i * 4 + 3] = 255;
  }

#if AIE_TUNED_AIE2
  // 32 pixels a step with no bor: zipping the bytes with themselves gives
  // (y, y) pairs and with 255 gives (y, 255) pairs; zipping those pairs gives
  // (y, y, y, 255). Four shuffles for four 256-bit stores. A row shorter than
  // four steps takes the plain loop; see rgba2gray.cc. A width that is not a
  // multiple of 32 finishes 16 pixels at a time as before.
  const v64uint8 alpha = ::aie::broadcast<uint8, 64>(255);
  for (int i = 0; i < height; i++) {
    const int steps = width / 32;
    auto body = [&]() __attribute__((always_inline)) {
      v64uint8 y = ::aie::load_v<32>(y_in).template grow<64>();
      y_in += 32;
      v64uint8 yy = shuffle(y, y, INTLV_lo_8o16);
      v64uint8 ya = shuffle(y, alpha, INTLV_lo_8o16);
      ::aie::store_v(rgba_out,
                     ::aie::vector<uint8, 64>(shuffle(yy, ya, INTLV_lo_16o32)));
      rgba_out += 64;
      ::aie::store_v(rgba_out,
                     ::aie::vector<uint8, 64>(shuffle(yy, ya, INTLV_hi_16o32)));
      rgba_out += 64;
    };
    if (steps >= 4) {
      AIE_LOOP_NO_UNROLL
      AIE_LOOP_MIN_ITERATION_COUNT(4)
      for (int j = 0; j < steps; j++)
        body();
    } else {
      AIE_LOOP_NO_UNROLL
      for (int j = 0; j < steps; j++)
        body();
    }
    for (int j = steps * 32; j < width; j += 16) {
      ::aie::vector<uint8, 16> data_buf = ::aie::load_v<16>(y_in);
      y_in += 16;

      ::aie::vector<uint8, 64> out = vector_broadcast(data_buf);

      v64uint8 fout = bor(out, alpha255);

      ::aie::store_v(rgba_out, ::aie::vector<uint8, 64>(fout));
      rgba_out += 64;
    }
  }
#else
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j += 16) {
      ::aie::vector<uint8, 16> data_buf = ::aie::load_v<16>(y_in);
      y_in += 16;

      // vector shuffle
      ::aie::vector<uint8, 64> out = vector_broadcast(data_buf);

      // bitwise OR with alpha value
      v64uint8 fout = bor(out, alpha255);

      ::aie::store_v(rgba_out, ::aie::vector<uint8, 64>(fout));
      rgba_out += 64;
    }
#endif

  event1();
  return;
  ;
}

void gray2rgba_aie_scalar(uint8_t *y_in, uint8_t *rgba_out,
                          const int32_t height, const int32_t width) {
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      uint8_t value = y_in[i * width + j];
      rgba_out[i * width * 4 + j * 4] = value;
      rgba_out[i * width * 4 + j * 4 + 1] = value;
      rgba_out[i * width * 4 + j * 4 + 2] = value;
      rgba_out[i * width * 4 + j * 4 + 3] = 255;
    }

  return;
  ;
}

extern "C" {

void gray2rgbaLine(uint8_t *in, uint8_t *out, int32_t lineWidth) {
  gray2rgba_aie(in, out, 1, lineWidth);
}

void gray2rgbaTile(uint8_t *in, uint8_t *out, int32_t tileHeight,
                   int32_t tileWidth) {
  gray2rgba_aie(in, out, tileHeight, tileWidth);
}

} // extern "C"

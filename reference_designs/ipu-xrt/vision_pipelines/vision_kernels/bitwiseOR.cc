//===- bitwiseOR.cc ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

template <typename T, int N>
void bitwiseOR_aie_scalar(const T *in1, const T *in2, T *out,
                          const int32_t width, const int32_t height) {
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      out[i * width + j] = in1[i * width + j] | in2[i * width + j];
}

template <typename T, int N>
void bitwiseOR_aie(const T *src1, const T *src2, T *dst, const int32_t width,
                   const int32_t height) {

  for (int j = 0; j < width * height; j += N)
    chess_prepare_for_pipelining chess_loop_range(
        14, ) // loop_range(14) - loop : 1 cycle
    {
      ::aie::vector<T, N> in1 = ::aie::load_v<N>(src1);
      src1 += N;
      ::aie::vector<T, N> in2 = ::aie::load_v<N>(src2);
      src2 += N;
      ::aie::vector<T, N> out;

      out = ::aie::bit_or(in1, in2);

      ::aie::store_v(dst, out);
      dst += N;
    }
}

extern "C" {

#if BIT_WIDTH == 8
void bitwiseORLine(uint8_t *in1, uint8_t *in2, uint8_t *out,
                   int32_t lineWidth) {
  bitwiseOR_aie<uint8_t, 64>(in1, in2, out, lineWidth, 1);
}

void bitwiseORTile(uint8_t *in1, uint8_t *in2, uint8_t *out, int32_t tileHeight,
                   int32_t tileWidth) {
  bitwiseOR_aie<uint8_t, 64>(in1, in2, out, tileWidth, tileHeight);
}

#elif BIT_WIDTH == 16
void bitwiseORLine(int16_t *in1, int16_t *in2, int16_t *out,
                   int32_t lineWidth) {
  bitwiseOR_aie<int16_t, 32>(in1, in2, out, lineWidth, 1);
}

void bitwiseORTile(int16_t *in1, int16_t *in2, int16_t *out, int32_t tileHeight,
                   int32_t tileWidth) {
  bitwiseOR_aie<int16_t, 32>(in1, in2, out, tileWidth, tileHeight);
}

#else // 32

void bitwiseORLine(int32_t *in1, int32_t *in2, int32_t *out,
                   int32_t lineWidth) {
  bitwiseOR_aie<int32_t, 16>(in1, in2, out, lineWidth);
}

void bitwiseORTile(int32_t *in1, int32_t *in2, int32_t *out, int32_t tileHeight,
                   int32_t tileWidth) {
  bitwiseOR_aie<int32_t, 16>(in1, in2, out, tileWidth, tileHeight);
}

#endif
} // extern "C"

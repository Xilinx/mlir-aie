//===- rgba2gray.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

const int32_t SRS_SHIFT = 15;
__attribute__((inline)) void xf_extract_rgb(uint8_t* ptr_rgba,
                                            ::aie::vector<uint8_t, 32>& r,
                                            ::aie::vector<uint8_t, 32>& g,
                                            ::aie::vector<uint8_t, 32>& b) {
    ::aie::vector<uint8_t, 32> rgba_channel0, rgba_channel1, rgba_channel3, rgba_channel2;
    rgba_channel0 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;
    rgba_channel1 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;
    rgba_channel2 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;
    rgba_channel3 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;

    // Unzip the interleaved channels
    auto[rg_temp, ba_temp] = ::aie::interleave_unzip(::aie::concat(rgba_channel0, rgba_channel1),
                                                     ::aie::concat(rgba_channel2, rgba_channel3), 2);
    r = ::aie::filter_even(rg_temp, 1);
    g = ::aie::filter_odd(rg_temp, 1);
    b = ::aie::filter_even(ba_temp, 1);
}

__attribute__((noinline)) void rgba2gray_aie(uint8_t *rgba_in, uint8_t *y_out, const int32_t height, const int32_t width) {
    //::aie::vector<int16_t, 16> WT(66, 129, 25, 128); //Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    //::aie::vector<int16_t, 16> WT(25, 129, 66, 128); //Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    ::aie::vector<int16_t, 16> WT((int16_t)round(0.299*(1<<SRS_SHIFT)), (int16_t)round(0.587*(1<<SRS_SHIFT)), (int16_t)round(0.114*(1<<SRS_SHIFT)), (1<<(SRS_SHIFT-1))); //Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    ::aie::vector<uint8_t, 32> c1 = ::aie::broadcast<uint8_t, 32>(1);
    ::aie::vector<uint8_t, 32> r, g, b;
    ::aie::vector<uint8_t, 32> y;

    for (int j = 0; (j < (width*height) / 32); j += 1) chess_prepare_for_pipelining {
        xf_extract_rgb(rgba_in, r, g, b);

        ::aie::accum<acc32, 32> acc;
        acc = ::aie::accumulate<32>(WT, 0, r, g, b, c1);
        y = acc.template to_vector<uint8_t>(SRS_SHIFT);

        ::aie::store_v(y_out, y);
        rgba_in += 128;
        y_out += 32;
    }
}

void rgba2gray_aie_scalar(uint8_t *rgba_in, uint8_t *y_out, const int32_t height, const int32_t width) {
    ///Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    const int colorMatrix[4] = {(int)round(0.299*65536),(int)round(0.587*65536),(int)round(0.114*65536), (65536/2)}; 
    for (int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            int r = (int) rgba_in[i*width*4 + j*4];
            int g = (int) rgba_in[i*width*4 + j*4 + 1];
            int b = (int) rgba_in[i*width*4 + j*4 + 2];
            int tmpSum = (colorMatrix[0]*r + colorMatrix[1]*g + colorMatrix[2]*b + colorMatrix[3]) >> 16;
            y_out[i*width+j] = (uint8_t)tmpSum; 

        }
    
    return;


}

extern "C" {

void rgba2grayLine(uint8_t *in, uint8_t *out, int32_t lineWidth) 
{
    rgba2gray_aie(in, out, 1, lineWidth);
}

void rgba2grayTile(uint8_t *in, uint8_t *out, int32_t tileHeight, int32_t tileWidth)
{
     rgba2gray_aie(in, out, tileHeight, tileWidth);
}

} // extern "C"

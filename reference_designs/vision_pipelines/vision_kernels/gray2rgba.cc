//===- gray2rgba.cc -------------------------------------------*- C++ -*-===//
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

::aie::vector<uint8,64> vector_broadcast(::aie::vector<uint8,16> e)
{
    v64uint8 lli = e.template grow<64>();
    lli = shuffle(lli, lli, T8_2x64_lo);
    lli = shuffle(lli, lli, T8_2x64_lo);
    return ::aie::vector<uint8,64>(lli);
}


void gray2rgba_aie(uint8_t *y_in, uint8_t *rgba_out, const int32_t height, const int32_t width) 
{
    // Initialize alpha vector
    ::aie::vector<uint8,64> alpha255 = ::aie::zeros<uint8,64>();
    for(int i=0; i<16; i++) {
        alpha255[i*4+3] = 255;
    }

    for (int i = 0; i < height; i++)
        for(int j = 0; j < width; j+=16)
        {
            ::aie::vector<uint8, 16> data_buf = ::aie::load_v<16>(y_in);
            y_in += 16;

            // vector shuffle
            ::aie::vector<uint8,64> out = vector_broadcast(data_buf);

            // bitwise OR with alpha value
            v64uint8 fout = bor(out, alpha255);

            ::aie::store_v(rgba_out, ::aie::vector<uint8,64>(fout));
            rgba_out += 64;
        }
    
    return;;
}

void gray2rgba_aie_scalar(uint8_t *y_in, uint8_t *rgba_out, const int32_t height, const int32_t width) {
    for (int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            uint8_t value = y_in[i*width+j];
            rgba_out[i*width*4 + j*4] = value;
            rgba_out[i*width*4 + j*4 + 1] = value;
            rgba_out[i*width*4 + j*4 + 2] = value;
            rgba_out[i*width*4 + j*4 + 3] = 255;
        }
    
    return;;
}

extern "C" {

void gray2rgbaLine(uint8_t *in, uint8_t *out, int32_t lineWidth) 
{
    gray2rgba_aie(in, out, 1, lineWidth);
}

void gray2rgbaTile(uint8_t *in, uint8_t *out, int32_t tileHeight, int32_t tileWidth)
{
     gray2rgba_aie(in, out, tileHeight, tileWidth);
}

} // extern "C"

//===- add_weighted.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

//#define __AIENGINE__ 1
#define NOCPP


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

//#include <imgproc/xf_addweighted_aie.hpp> // NOTE: use of float2fix not supported in aie2
#include <aie_api/aie.hpp>


const int32_t SRS_SHIFT = 14;

template <typename T, int N, int MAX>
void addweighted_aie_scalar(const T* in1, const T*  in2, T* out, 
                        const int32_t width, const int32_t height,
                        const int16_t alpha, const int16_t beta, const T gamma) {
    for (int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            int tmpIn1 = in1[i*width+j]*alpha;
            int tmpIn2 = in2[i*width+j]*beta;
            int tmp = ((tmpIn1 + tmpIn2 +  (1<<(SRS_SHIFT-1))) >> SRS_SHIFT) + gamma;
            tmp = (tmp > MAX) ? MAX : (tmp < 0) ? 0 : tmp; //saturate
            out[i*width+j] = (T)tmp;
        }

}

template <typename T, int N, int MAX>
void addweighted_aie(const T* src1, const T*  src2, T* dst, 
                        const int32_t width, const int32_t height,
                        const int16_t alphaFixedPoint, const int16_t betaFixedPoint, const T gamma) {
    
    ::aie::set_saturation(aie::saturation_mode::saturate); // Needed to saturate properly to uint8

    ::aie::vector<int16_t, N> coeff(alphaFixedPoint, betaFixedPoint);
    ::aie::vector<T, N> gamma_coeff;
    ::aie::accum<acc32, N> gamma_acc;
    for (int i = 0; i < N; i++) {
        gamma_coeff[i] = gamma;
    }
    gamma_acc.template from_vector(gamma_coeff, 0);
    for (int j = 0; j < width * height; j += N)             // 16 samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) // loop_range(14) - loop : 1 cycle
        {
            ::aie::vector<T, N> data_buf1 = ::aie::load_v<N>(src1);
            src1 += N;
            ::aie::vector<T, N> data_buf2 = ::aie::load_v<N>(src2);
            src2 += N;
            ::aie::accum<acc32, N> acc = ::aie::accumulate<N>(
                gamma_acc, coeff, 0, data_buf1, data_buf2); // weight[0] * data_buf1 + weight[1] * data_buf2
            ::aie::store_v(dst, acc.template to_vector<T>(SRS_SHIFT));
            dst += N;
        }        
}

extern "C" {

#if BIT_WIDTH == 8
void addWeightedLine(uint8_t *in1, uint8_t *in2, uint8_t *out, int32_t lineWidth, int16_t alpha, int16_t beta, uint8_t gamma) {
    addweighted_aie<uint8_t, 32, UINT8_MAX>(in1, in2, out, lineWidth, 1, alpha, beta, gamma);
}

void addWeightedTile(uint8_t *in1, uint8_t *in2, uint8_t *out, int32_t tileHeight, int32_t tileWidth, int16_t alpha, int16_t beta, uint8_t gamma) {
    addweighted_aie<uint8_t, 32, UINT8_MAX>(in1, in2, out, tileWidth, tileHeight, alpha, beta, gamma);
}

#elif BIT_WIDTH == 16
void addWeightedLine(int16_t *in1, int16_t *in2, int16_t *out, int32_t lineWidth, int16_t alpha, int16_t beta, int16_t gamma) {
    addweighted_aie<int16_t, 16, INT16_MAX>(in1, in2, out, lineWidth, 1, alpha, beta, gamma);
}

void addWeightedTile(int16_t *in1, int16_t *in2, int16_t *out, int32_t tileHeight, int32_t tileWidth, int16_t alpha, int16_t beta, int16_t gamma) {
    addweighted_aie<int16_t, 16, INT16_MAX>(in1, in2, out, tileWidth, tileHeight, alpha, beta, gamma);
}

#else // 32

void addWeightedLine(int32_t *in1, int32_t *in2, int32_t *out, int32_t lineWidth, int16_t alpha, int16_t beta, int32_t gamma) {
    addweighted_aie<int32_t, 16, INT32_MAX>(in1, in2, out, lineWidth, 1, alpha, beta, gamma);
}

void addWeightedTile(int32_t *in1, int32_t *in2, int32_t *out, int32_t tileHeight, int32_t tileWidth, int16_t alpha, int16_t beta, int32_t gamma) {
    addweighted_aie<int32_t, 16, INT32_MAX>(in1, in2, out, tileWidth, tileHeight, alpha, beta, gamma);
}

#endif
} // extern "C"

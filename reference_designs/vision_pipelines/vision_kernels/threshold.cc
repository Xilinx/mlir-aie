//===- threshold.cc ----------------------------------------------*- C++ -*-===//
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

enum _threshold_type {
    XF_THRESHOLD_TYPE_BINARY = 0,
    XF_THRESHOLD_TYPE_BINARY_INV = 1,
    XF_THRESHOLD_TYPE_TRUNC = 2,
    XF_THRESHOLD_TYPE_TOZERO = 3,
    XF_THRESHOLD_TYPE_TOZERO_INV = 4,
};

//#define THRESH_TYPE XF_THRESHOLD_TYPE_BINARY

#include <aie_api/aie.hpp>

template <typename T, int N>
__attribute__((noinline)) void threshold_aie(
    T* img_in, T* img_out, const int32_t img_width, const int32_t img_height, const T& thresh_val, const T& max_val, const uint8_t thresholdType) {
    ::aie::vector<T, N> constants;
    ::aie::vector<T, N> data_out;
    ::aie::mask<N> temp_val;
    constants[0] = 0;          // updating constant zero_val value
    constants[1] = thresh_val; // updating constant threshold value
    constants[2] = max_val;    // updating constant max_val value

    switch (thresholdType) {
        case XF_THRESHOLD_TYPE_TRUNC:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                data_out = ::aie::min(constants[1], data_buf1);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(constants[0], constants[2], temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY_INV:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(constants[2], constants[0], temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(constants[0], data_buf1, temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO_INV:
           for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(data_buf1, constants[0], temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        default:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                data_out = ::aie::min(constants[1], data_buf1);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }


    }
}

template <typename T, int N>
__attribute__((noinline)) void threshold4Ch_aie(
    T* img_in, T* img_out, const int32_t img_width, const int32_t img_height, const T& thresh_val1, const T& thresh_val2, const T& thresh_val3, const T& thresh_val4, const T& max_val1, const T& max_val2, const T& max_val3, const T& max_val4, const uint8_t thresholdType) {
    ::aie::vector<T, N> constants;
    ::aie::vector<T, N> data_out;
    ::aie::mask<N> temp_val;
    // constants[0] = 0;          // updating constant zero_val value
    // constants[1] = thresh_val; // updating constant threshold value
    // constants[2] = max_val;    // updating constant max_val value

    ::aie::vector<T, N> mask_zeros  = ::aie::zeros<T, N>();
    ::aie::vector<T, N> mask_thresh;
    ::aie::vector<T, N> mask_max;
    for(int i=0; i<N/4; i++) {
        mask_thresh[i*4]   = thresh_val1;
        mask_thresh[i*4+1] = thresh_val2;
        mask_thresh[i*4+2] = thresh_val3;
        mask_thresh[i*4+3] = thresh_val4;
        mask_max[i*4]      = max_val1;
        mask_max[i*4+1]    = max_val2;
        mask_max[i*4+2]    = max_val3;
        mask_max[i*4+3]    = max_val4;
    }

    switch (thresholdType) {
        case XF_THRESHOLD_TYPE_TRUNC:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                data_out = ::aie::min(mask_thresh, data_buf1);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(mask_zeros, mask_max, temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY_INV:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(mask_max, mask_zeros, temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(mask_zeros, data_buf1, temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO_INV:
           for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(data_buf1, mask_zeros, temp_val);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }
            break;
        default:
            for (int j = 0; j < (img_height * img_width); j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) { 
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(img_in); // in:00++15|_________|_________|_________
                img_in += N;
                data_out = ::aie::min(mask_thresh, data_buf1);
                ::aie::store_v(img_out, data_out);
                img_out += N;
            }


    }
}


extern "C" {

#if BIT_WIDTH == 8

void threshold(uint8_t *img_in, uint8_t *img_out, int32_t thresh_val, int32_t max_val, 
               int32_t img_width, int32_t img_height) 
{
	threshold_aie<uint8_t, 64>(img_in, img_out, img_width, img_height, thresh_val, max_val, XF_THRESHOLD_TYPE_BINARY);
}

void thresholdTile(uint8_t *in, uint8_t *out, int32_t tileHeight, int32_t tileWidth, uint8_t thresholdValue, uint8_t maxValue, uint8_t thresholdType){
    threshold_aie<uint8_t, 64>(in, out, tileWidth, tileHeight, thresholdValue, maxValue, thresholdType);
}

void thresholdLine(uint8_t *in, uint8_t *out, int32_t lineWidth, uint8_t thresholdValue, uint8_t maxValue, uint8_t thresholdType) {
    threshold_aie<uint8_t, 64>(in, out, lineWidth, 1, thresholdValue, maxValue, thresholdType);
} 

void threshold4ChLine(uint8_t *in, uint8_t *out, int32_t lineWidth, uint8_t thresholdValue1, uint8_t thresholdValue2, uint8_t thresholdValue3, uint8_t thresholdValue4, uint8_t maxValue1, uint8_t maxValue2, uint8_t maxValue3, uint8_t maxValue4, uint8_t thresholdType) {
    threshold4Ch_aie<uint8_t, 64>(in, out, lineWidth, 1, thresholdValue1, thresholdValue2, thresholdValue3, thresholdValue4, maxValue1, maxValue2, maxValue3, maxValue4, thresholdType);
} 

#elif BIT_WIDTH == 16

void threshold(int16_t *img_in, int16_t *img_out, int32_t thresh_val, int32_t max_val, 
               int32_t img_width, int32_t img_height) 
{
	threshold_aie<int16_t, 32>(img_in, img_out, img_width, img_height, thresh_val, max_va, XF_THRESHOLD_TYPE_BINARY);
}

void thresholdTile(int16_t *in, int16_t *out, int32_t tileHeight, int32_t tileWidth, int16_t thresholdValue, int16_t maxValue, uint8_t thresholdType){
    threshold_aie<int16_t, 32>(in, out, tileWidth, tileHeight, thresholdValue, maxValue), thresholdType;
}

void thresholdLine(int16_t *in, int16_t *out, int32_t lineWidth, int16_t thresholdValue, int16_t maxValue, uint8_t thresholdType) {
    threshold_aie<int16_t, 32>(in, out, lineWidth, 1, thresholdValue, maxValue, thresholdType);
} 

#else // 32

void threshold(int32_t *img_in, int32_t *img_out, int32_t thresh_val, int32_t max_val, 
               int32_t img_width, int32_t img_height) 
{
	threshold_aie<int32_t, 16>(img_in, img_out, img_width, img_height, thresh_val, max_val, XF_THRESHOLD_TYPE_BINARY);
}

void thresholdTile(int32_t *in, int32_t *out, int32_t tileHeight, int32_t tileWidth, int32_t thresholdValue, int32_t maxValue, uint8_t thresholdType){
    threshold_aie<int32_t, 16>(in, out, tileWidth, tileHeight, thresholdValue, maxValue, thresholdType);
}

void thresholdLine(int32_t *in, int32_t *out, int32_t lineWidth, int32_t thresholdValue, int32_t maxValue, uint8_t thresholdType) {
    threshold_aie<int32_t, 16>(in, out, lineWidth, 1, thresholdValue, maxValue, thresholdType);
} 

#endif

} // extern "C"

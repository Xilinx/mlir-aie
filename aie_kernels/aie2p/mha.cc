//===- mha.cc ---------------------------*- C++-----*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===-----------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#define ROUNDING_MODE aie::rounding_mode::conv_even

extern "C" {
    void matmul_scalar_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
    void matmul_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
    void matmul_bf16_bf16_rowmaj(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out);
    void partial_softmax_bf16(bfloat16 *input, bfloat16 *output, float *scale_buffer, const int32_t input_size, const int32_t row_idx, const int32_t row_size);
    void passThroughLine(int16_t *in, int16_t *out, int32_t lineWidth);

    // VJUNG: Regular PassThroughLine crashes on aie211
    void passThroughLineScalar(float *in, float *out, int32_t lineWidth) {

        ::aie::set_rounding(ROUNDING_MODE);

        for (int32_t i = 0; i < lineWidth; i++) {
            out[i] = in[i];
        }
    }

    void passThroughLineScalarDebug(bfloat16 *in, bfloat16 *out, int32_t lineWidth) {

        ::aie::set_rounding(ROUNDING_MODE);

        for (int32_t i = 0; i < lineWidth; i++) {
            out[i] = in[i];
        }
    }

    void matmul_bf16_bf16_wrapper(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
        
        ::aie::set_rounding(ROUNDING_MODE);
        matmul_bf16_bf16(a_in, b_in, c_out);
    }

    void matmul_bf16_bf16_wrapper_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
        
        ::aie::set_rounding(ROUNDING_MODE);
        matmul_scalar_bf16_bf16(a_in, b_in, c_out);
    }

    void matmul_PV(bfloat16 *Q, bfloat16 *K, bfloat16 *out, float *scale_buffer, const int32_t B_q, int32_t first_iter) {
        
        // ::aie::set_saturation(aie::saturation_mode::saturate);
        ::aie::set_rounding(ROUNDING_MODE);

        // 16: O dims = [(4, 64), (4, 8), (2, 32), (8, 1)]
        // 64: O dims = [(16, 256), (4, 8), (8, 32), (8, 1)]
        // VJUNG: Scale O_{i-1} by 1/exp(m_{i-1} - m_{i}) store in scale_buffer[3*B_q:3*B_q + B_q]
        // VJUNG: Skip this for the first iteration as 1/exp(m_{i-1} - m_{i}) degenerates to inf due to m intizalized to -inf
        if (first_iter != 0) {
            for(int32_t l = 0; l < 16; l++){
                for(int32_t k = 0; k < 4; k++){ // Iterate for 4 rows 
                    for(int32_t j = 0; j < 8; j++){ // Each row is broken down into 2 blocks of 8
                        for (int32_t i = 0; i < 8; i++) {
                            out[i + j*32 + k*8 + l*256] = out[i + j*32 + k*8 + l*256] * scale_buffer[3*B_q + (k + l*4)];
                        }
                    }
                }
            }
        }
        
        matmul_bf16_bf16_rowmaj(Q, K, out);

    }


    void rescale_O(bfloat16 *O, float *scale_buffer, int32_t B_q) {

        ::aie::set_rounding(ROUNDING_MODE);

        // VJUNG: Only after all KV are processed
        // VJUNG: TODO: Make this generic for every tile size
        // VJUNG: Need to scale depending on the data layout at the output of GEMM
        // VJUNG: Scale O_{i} by 1/l_{i} 
        for(int32_t l = 0; l < 16; l++){
            for(int32_t k = 0; k < 4; k++){ // Iterate for 4 rows
                for(int32_t j = 0; j < 8; j++){ // Each row is broken down into 2 blocks of 8
                    for (int32_t i = 0; i < 8; i++) {
                        O[i + j*32 + k*8 + l*256] = O[i + j*32 + k*8 + l*256] * aie::inv(scale_buffer[2*B_q + (k + l*4)]);
                    }
                }
            }
        }
    }


    void partial_softmax(bfloat16 *A, bfloat16 *P, float *scale_buffer, float inv_scale, int32_t B_q, int32_t B_kv) {

        ::aie::set_rounding(ROUNDING_MODE);

        for (int32_t i = 0; i < B_q * B_kv; i++) {
            A[i] = A[i] * inv_scale;
        }
        for (int32_t i = 0; i < B_q; i++) {
            partial_softmax_bf16(A + B_kv*i, P + B_kv*i, scale_buffer, B_kv, i, B_q);
        }
    }

    void init_scale_buffer(float *scale_buffer, int32_t size) {
        // VJUNG: TODO: Vectorize
        ::aie::set_rounding(ROUNDING_MODE);

        // VJUNG: m_{i-1} vector
        for (int32_t i = 0; i < size; i++) {
            scale_buffer[i] = std::numeric_limits<float>::lowest();
        }
        // VJUNG: m_{i} vector
        for (int32_t i = 0; i < size; i++) {
            scale_buffer[i + size] = std::numeric_limits<float>::lowest();
        }
        // VJUNG: l_{i} vector
        for (int32_t i = 0; i < size; i++) {
            scale_buffer[i + 2*size] = 0.0f;
        }

        // for (int32_t i = 0; i < size; i++) {
        //     scale_buffer[i + 3*size] = 0.0f;
        // }
    }
}
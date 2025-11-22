// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "aie_kernel_utils.h"

#include <aie_api/aie.hpp>

void matvec_scalar(uint32_t m, uint32_t k, uint32_t row_offset, bfloat16 *a, bfloat16 *b, bfloat16 *c)
{
    for (uint32_t row = 0; row < m; row++) {
        float acc = 0;
        for (uint32_t i = 0; i < k; i++) {
            acc += a[row * k + i] * b[i];
        }
        c[row + row_offset * m] = static_cast<bfloat16>(acc);
    }
}

template <uint32_t r>
void matvec_vectorized(uint32_t m,
                       uint32_t k,
                       uint32_t row_offset,
                       const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b,
                       bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    c += row_offset * m;
    bfloat16 *c_end = c + m;
    const bfloat16 *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum acc = aie::zeros<accfloat, r>();
        // The following two pragmas enable pipelining the zero-overhead loop, but they do assume that k is at least
        // two. This assumption should hold for any useful use of this function; if k were one, this would be a simple
        // scalar multiplication of a vector.
        AIE_LOOP_MIN_ITERATION_COUNT(2)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

extern "C" {

void matvec_scalar_bf16_bf16(uint32_t m,
                             uint32_t k,
                             uint32_t row_offset,
                             bfloat16 *a_in,
                             bfloat16 *b_in,
                             bfloat16 *c_out)
{
    matvec_scalar(m, k, row_offset, a_in, b_in, c_out);
}

void matvec_vectorized_bf16_bf16(uint32_t m,
                                 uint32_t k,
                                 uint32_t row_offset,
                                 bfloat16 *a_in,
                                 bfloat16 *b_in,
                                 bfloat16 *c_out)
{
    matvec_vectorized<64>(m, k, row_offset, a_in, b_in, c_out);
}

} // extern "C"
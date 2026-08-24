//===- mv.cc ----------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif

void matvec_scalar(uint32_t m,
                   uint32_t k,
                   const bfloat16 *__restrict a,
                   const bfloat16 *__restrict b,
                   bfloat16 *__restrict c)
{
    for (uint32_t row = 0; row < m; row++) {
        float acc = 0;
        for (uint32_t i = 0; i < k; i++) {
            acc += a[row * k + i] * b[i];
        }
        c[row] = static_cast<bfloat16>(acc);
    }
}

/*
Matrix-vector multiplication kernel

 - m: Number of output rows == number of rows in the input matrix
 - k: Number of columns in the input matrix == length of the input vector
 - a: Pointer to the input matrix, stored in row-major order
 - b: Pointer to the input vector
 - c: Pointer to the output vector
 - r: Vector size; data from the matrix and vector will be loaded in and processed in chunks of this size
*/
template <uint32_t r, uint32_t k>
void matvec_vectorized(uint32_t m, const bfloat16 *__restrict a, const bfloat16 *__restrict b, bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *c_end = c + m;
    const bfloat16 *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum acc = aie::zeros<accfloat, r>();
        // The following two pragmas enable pipelining the zero-overhead loop, but they do assume that there are at
        // least two iterations of the loop, i.e. k >= 2*r. This pragma will break the code if that is not the case!
        AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

extern "C" {

/* The row offset parameter in the functions below is a workaround. The output will be written to c + row_offset * m.
 * This is simpler than to do pointer arithmetic in the calling MLIR code, but that's all this is for -- an offset into
 * `c`.  */

void matvec_scalar_bf16_bf16(uint32_t m,
                             uint32_t row_offset,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_scalar(m, DIM_K, a_in, b_in, c_out);
}

void matvec_vectorized_bf16_bf16(uint32_t m,
                                 uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_vectorized<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
}

} // extern "C"

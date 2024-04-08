//===- dequant.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "idct_8x8_mmult.h"

alignas(v16int16) int16_t
    dequant_lut[DCT8x8_BLOCK_WIDTH * DCT8x8_BLOCK_HEIGHT] = {
        8,  6,  5,  8,  12, 20, 26, 31, 6,  6,  7,  10, 13, 29, 30, 28,
        7,  7,  8,  12, 20, 29, 35, 28, 7,  9,  11, 15, 26, 44, 40, 31,
        9,  11, 19, 28, 34, 55, 52, 39, 12, 18, 28, 32, 41, 52, 57, 46,
        25, 32, 39, 44, 52, 61, 60, 51, 36, 46, 48, 49, 56, 50, 52, 50};

void extern "C" dequant_8x8(int16_t *restrict input, int16_t *restrict output) {
  // int16_t *restrict new_ptr_in = (int16_t *) input;
  // int16_t *restrict new_ptr_out = (int16_t *) output;

  v16int16 *restrict ptr_in = (v16int16 *)input;
  v16int16 *restrict ptr_out = (v16int16 *)output;

  v16int16 *restrict ptr_dq = (v16int16 *)dequant_lut;

  v16int16 dequant, in_lo_0, in_lo_1, in_hi_0, in_hi_1;
  v16acc48 dq_inp_0, dq_inp_1;

  in_lo_0 = *ptr_in++;
  in_lo_1 = *ptr_in++;

  dequant = *ptr_dq++;

#if (NUM_DCT8x8_BLOCKS_PER_ITERATION > 1)
  for (int j = 0; j < NUM_DCT8x8_BLOCKS_PER_ITERATION; ++j)
    chess_prepare_for_pipelining chess_loop_range(
        NUM_DCT8x8_BLOCKS_PER_ITERATION, ) // pre-condition n > 1
#endif
    {
      dq_inp_0 = mul(in_lo_0, dequant);
      *ptr_out++ = srs(dq_inp_0, 0);
      in_hi_0 = *ptr_in++;

      dequant = *ptr_dq++;
      dq_inp_1 = mul(in_lo_1, dequant);
      *ptr_out++ = srs(dq_inp_1, 0);
      in_hi_1 = *ptr_in++;

      dequant = *ptr_dq++;
      dq_inp_0 = mul(in_hi_0, dequant);
      *ptr_out++ = srs(dq_inp_0, 0);

      dequant = *ptr_dq++;
      dq_inp_1 = mul(in_hi_1, dequant);
      *ptr_out++ = srs(dq_inp_1, 0);

#if (NUM_DCT8x8_BLOCKS_PER_ITERATION > 1)
      // load data for next iteration
      in_lo_0 = *ptr_in++;
      in_lo_1 = *ptr_in++;
      ptr_dq = (v16int16 *)dequant_lut;
      dequant = *ptr_dq++;
#endif
    }
}
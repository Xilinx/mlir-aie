//===- idct_8x8_mmult.h -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef __IDCT_8x8_MMULT_H__
#define __IDCT_8x8_MMULT_H__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// #include <adf.h>

#define DCT8x8_BLOCK_WIDTH (8)
#define DCT8x8_BLOCK_HEIGHT (8)
#define NUM_DCT8x8_BLOCKS_PER_ITERATION (1)
#define NUM_ITERATION (1)
#define BYTES_PER_DATA (2)

#define DCT8x8_BUF_SIZE                                                        \
  (NUM_DCT8x8_BLOCKS_PER_ITERATION * DCT8x8_BLOCK_WIDTH * DCT8x8_BLOCK_HEIGHT)

#define DCT8x8_SHIFT_H_TXFM (11) // 13 - 2
#define DCT8x8_SHIFT_V_TXFM (18) // 13 + 5

#define c1 11363
#define c2 10703
#define c3 9633
#define c4 8192
#define c5 6436
#define c6 4433
#define c7 2260

#define _c1 -11363
#define _c2 -10703
#define _c3 -9633
#define _c4 -8192
#define _c5 -6436
#define _c6 -4433
#define _c7 -2260

extern "C" {
void dequant_8x8(int16_t *restrict input, int16_t *restrict output);
void idct_8x8_mmult_h(int16_t *restrict input, int16_t *restrict output);
void idct_8x8_mmult_v(int16_t *restrict input, int16_t *restrict output);
}

#endif
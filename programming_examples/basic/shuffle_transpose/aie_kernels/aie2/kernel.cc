//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 AMD Inc.

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include <stdio.h>

#define byte_incr(in_ptr, strides_bytes) (in_ptr + strides_bytes)

#define LOAD_16x16_v4(iv1, iv2, iv3, iv4, in_ptr, offset, stride)              \
  {                                                                            \
                                                                               \
    iv1 = aie::load_v<64>((uint8_t *)in_ptr + offset);                         \
    iv2 = aie::load_v<64>((uint8_t *)in_ptr + stride + offset);                \
    iv3 = aie::load_v<64>((uint8_t *)in_ptr + (2 * stride) + offset);          \
    iv4 = aie::load_v<64>((uint8_t *)in_ptr + (3 * stride) + offset);          \
  }

// The third parameter to `shuffle` specifies the shuffle mode
// shuffle mode: ${DataWidth}_${ShuffleType}_${hi|lo}
#define TRANSPOSE_16x16_1B(iv1, iv2, iv3, iv4, tk_16_4_v1, tk_16_4_v2,         \
                           tk_16_4_v3, tk_16_4_v4, tk_8_8_v1_lo, tk_8_8_v1_hi, \
                           tk_8_8_v2_lo, tk_8_8_v2_hi, tk_4_16_v1_lo,          \
                           tk_4_16_v1_hi, tk_4_16_v2_lo, tk_4_16_v2_hi)        \
  {                                                                            \
    tk_16_4_v1 = shuffle(iv1, iv1, T8_4x16);                                   \
    tk_16_4_v2 = shuffle(iv2, iv2, T8_4x16);                                   \
    tk_16_4_v3 = shuffle(iv3, iv3, T8_4x16);                                   \
    tk_16_4_v4 = shuffle(iv4, iv4, T8_4x16);                                   \
                                                                               \
    tk_8_8_v1_lo = shuffle(tk_16_4_v1, tk_16_4_v2, T32_2x16_lo);               \
    tk_8_8_v1_hi = shuffle(tk_16_4_v1, tk_16_4_v2, T32_2x16_hi);               \
    tk_8_8_v2_lo = shuffle(tk_16_4_v3, tk_16_4_v4, T32_2x16_lo);               \
    tk_8_8_v2_hi = shuffle(tk_16_4_v3, tk_16_4_v4, T32_2x16_hi);               \
                                                                               \
    tk_4_16_v1_lo = shuffle(tk_8_8_v1_lo, tk_8_8_v2_lo, T64_2x8_lo);           \
    tk_4_16_v1_hi = shuffle(tk_8_8_v1_lo, tk_8_8_v2_lo, T64_2x8_hi);           \
    tk_4_16_v2_lo = shuffle(tk_8_8_v1_hi, tk_8_8_v2_hi, T64_2x8_lo);           \
    tk_4_16_v2_hi = shuffle(tk_8_8_v1_hi, tk_8_8_v2_hi, T64_2x8_hi);           \
  }

extern "C" {
void transpose_16x16(uint8_t *restrict inIter, uint8_t *restrict outIter);
}

void transpose_16x16(uint8_t *restrict inIter, uint8_t *restrict outIter) {

  aie::vector<uint8_t, 64> iv1, iv2, iv3, iv4, tk_16_4_v1, tk_16_4_v2,
      tk_16_4_v3, tk_16_4_v4, tk_8_8_v1_lo, tk_8_8_v1_hi, tk_8_8_v2_lo,
      tk_8_8_v2_hi, tk_4_16_v1_lo, tk_4_16_v1_hi, tk_4_16_v2_lo, tk_4_16_v2_hi;

  LOAD_16x16_v4(iv1, iv2, iv3, iv4, inIter, 0, 64);

  TRANSPOSE_16x16_1B(iv1, iv2, iv3, iv4, tk_16_4_v1, tk_16_4_v2, tk_16_4_v3,
                     tk_16_4_v4, tk_8_8_v1_lo, tk_8_8_v1_hi, tk_8_8_v2_lo,
                     tk_8_8_v2_hi, tk_4_16_v1_lo, tk_4_16_v1_hi, tk_4_16_v2_lo,
                     tk_4_16_v2_hi);

  aie::store_v(outIter, tk_4_16_v1_lo);
  outIter = byte_incr(outIter, 64);
  aie::store_v(outIter, tk_4_16_v1_hi);
  outIter = byte_incr(outIter, 64);
  aie::store_v(outIter, tk_4_16_v2_lo);
  outIter = byte_incr(outIter, 64);
  aie::store_v(outIter, tk_4_16_v2_hi);
}

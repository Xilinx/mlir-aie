//===- idct_horizontal.cc -------------------------------------------------*-
// C++ -*-===//
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
    cospi_coeff_lut_h[DCT8x8_BLOCK_WIDTH * DCT8x8_BLOCK_HEIGHT] = {
        c4,  c1,  c2,  c3,  c4,  c5,  c6,  c7,  c4,  c3,  c6,  _c7, _c4,
        _c1, _c2, _c5, c4,  c5,  _c6, _c1, _c4, c7,  c2,  c3,  c4,  c7,
        _c2, _c5, c4,  c3,  _c6, _c1, c4,  _c7, _c2, c5,  c4,  _c3, _c6,
        c1,  c4,  _c5, _c6, c1,  _c4, _c7, c2,  _c3, c4,  _c3, c6,  c7,
        _c4, c1,  _c2, c5,  c4,  _c1, c2,  _c3, c4,  _c5, c6,  _c7};

// alignas(v16int16) int16_t
// cospi_coeff_lut_h[DCT8x8_BLOCK_WIDTH*DCT8x8_BLOCK_HEIGHT] = {
//     1,  1,  1, 1,  1,  1,  1,  1,
//     1,  1,  1, 1,  1,  1,  1,  1,
//     1,  1,  1, 1,  1,  1,  1,  1,
//     1,  1,  1, 1,  1,  1,  1,  1,
//     1,  1,  1, 1,  1,  1,  1,  1,
//      1,  1,  1, 1,  1,  1,  1,  1,
//      1,  1,  1, 1,  1,  1,  1,  1,
//      1,  1,  1, 1,  1,  1,  1,  1
// };

// void idct_8x8_mmult_h(int16_t  *restrict input,
//                    int16_t *restrict output)
// {
//       for (int i = 0; i < 64; i ++){
//         output[i] = input[i];
//     }

// }

void idct_8x8_mmult_h(int16_t *restrict input, int16_t *restrict output) {
  v16int16 *restrict ptr_in = (v16int16 *)input;
  v8int16 *restrict ptr_out = (v8int16 *)output;
  v16int16 *restrict ptr_coeff = (v16int16 *)cospi_coeff_lut_h;

  v32int16 chess_storage(XA) vec_a_lo = undef_v32int16();
  v32int16 chess_storage(XB) vec_a_hi = undef_v32int16();

  v16int16 vec_b0, vec_b1, vec_b2, vec_b3;
  v8acc48 acc_lo, acc_hi;
  int16_t SHIFT_TXFM = DCT8x8_SHIFT_H_TXFM;

  set_rnd(rnd_pos_inf);
  vec_a_lo = upd_w(vec_a_lo, 0, *ptr_in++);
  vec_a_lo = upd_w(vec_a_lo, 1, *ptr_in++);

  vec_b0 = *ptr_coeff++;
  vec_b1 = *ptr_coeff++;
  vec_b2 = *ptr_coeff++;
  vec_b3 = *ptr_coeff++;

#if (NUM_DCT8x8_BLOCKS_PER_ITERATION > 1)
  for (int j = 0; j < NUM_DCT8x8_BLOCKS_PER_ITERATION; ++j)
    chess_prepare_for_pipelining chess_loop_range(
        NUM_DCT8x8_BLOCKS_PER_ITERATION, )
#endif
    {
      acc_lo =
          mul8(vec_a_lo, 0, 0x38303830, 2, 0x3210, vec_b0, 0, 0x88880000, 1);
      vec_a_hi = upd_w(vec_a_hi, 0, *ptr_in++);
      acc_lo = mac8(acc_lo, vec_a_lo, 0, 0x3a323a32, 2, 0x3210, vec_b0, 4,
                    0x88880000, 1);
      vec_a_hi = upd_w(vec_a_hi, 1, *ptr_in++);
      *ptr_out++ = srs(acc_lo, SHIFT_TXFM);

      acc_hi =
          mul8(vec_a_lo, 0, 0x38303830, 2, 0x3210, vec_b1, 0, 0x88880000, 1);
      acc_hi = mac8(acc_hi, vec_a_lo, 0, 0x3a323a32, 2, 0x3210, vec_b1, 4,
                    0x88880000, 1);
      *ptr_out++ = srs(acc_hi, SHIFT_TXFM);

      acc_lo =
          mul8(vec_a_lo, 0, 0x38303830, 2, 0x3210, vec_b2, 0, 0x88880000, 1);
      acc_lo = mac8(acc_lo, vec_a_lo, 0, 0x3a323a32, 2, 0x3210, vec_b2, 4,
                    0x88880000, 1);
      *ptr_out++ = srs(acc_lo, SHIFT_TXFM);

      acc_hi =
          mul8(vec_a_lo, 0, 0x38303830, 2, 0x3210, vec_b3, 0, 0x88880000, 1);
      acc_hi = mac8(acc_hi, vec_a_lo, 0, 0x3a323a32, 2, 0x3210, vec_b3, 4,
                    0x88880000, 1);
      *ptr_out++ = srs(acc_hi, SHIFT_TXFM);

      acc_lo =
          mul8(vec_a_hi, 0, 0x38303830, 2, 0x3210, vec_b0, 0, 0x88880000, 1);
      acc_lo = mac8(acc_lo, vec_a_hi, 0, 0x3a323a32, 2, 0x3210, vec_b0, 4,
                    0x88880000, 1);
      *ptr_out++ = srs(acc_lo, SHIFT_TXFM);

      acc_hi =
          mul8(vec_a_hi, 0, 0x38303830, 2, 0x3210, vec_b1, 0, 0x88880000, 1);
      acc_hi = mac8(acc_hi, vec_a_hi, 0, 0x3a323a32, 2, 0x3210, vec_b1, 4,
                    0x88880000, 1);
      *ptr_out++ = srs(acc_hi, SHIFT_TXFM);

      acc_lo =
          mul8(vec_a_hi, 0, 0x38303830, 2, 0x3210, vec_b2, 0, 0x88880000, 1);
      acc_lo = mac8(acc_lo, vec_a_hi, 0, 0x3a323a32, 2, 0x3210, vec_b2, 4,
                    0x88880000, 1);
      *ptr_out++ = srs(acc_lo, SHIFT_TXFM);

      acc_hi =
          mul8(vec_a_hi, 0, 0x38303830, 2, 0x3210, vec_b3, 0, 0x88880000, 1);
      acc_hi = mac8(acc_hi, vec_a_hi, 0, 0x3a323a32, 2, 0x3210, vec_b3, 4,
                    0x88880000, 1);
      *ptr_out++ = srs(acc_hi, SHIFT_TXFM);

#if (NUM_DCT8x8_BLOCKS_PER_ITERATION > 1)
      vec_a_lo = upd_w(vec_a_lo, 0, *ptr_in++);
      vec_a_lo = upd_w(vec_a_lo, 1, *ptr_in++);
#endif
    }
}
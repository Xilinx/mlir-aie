/*  (c) Copyright 2019-2022 Xilinx, Inc. All rights reserved.
    (c) Copyright 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

    This file contains confidential and proprietary information
    of Xilinx, Inc. and is protected under U.S. and
    international copyright and other intellectual int16_t
    laws.

    DISCLAIMER
    This disclaimer is not a license and does not grant any
    rights to the materials distributed herewith. Except as
    otherwise provided in a valid license issued to you by
    Xilinx, and to the maximum extent permitted by applicable
    law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
    WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
    AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
    BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
    INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
    (2) Xilinx shall not be liable (whether in contract or tort,
    including negligence, or under any other theory of
    liability) for any loss or damage of any kind or nature
    related to, arising under or in connection with these
    materials, including for any direct, or any indirect,
    special, incidental, or consequential loss or damage
    (including loss of data, profits, goodwill, or any type of
    loss or damage suffered as a result of any action brought
    by a third party) even if such damage or loss was
    reasonably foreseeable or Xilinx had been advised of the
    possibility of the same.

    CRITICAL APPLICATIONS
    Xilinx products are not designed or intended to be fail-
    safe, or for use in any application requiring fail-safe
    performance, such as life-support or safety devices or
    systems, Class III medical devices, nuclear facilities,
    applications related to the deployment of airbags, or any
    other applications that could lead to death, personal
    injury, or severe int16_t or environmental damage
    (individually and collectively, "Critical
    Applications"). Customer assumes the sole risk and
    liability of any use of Xilinx products in Critical
    Applications, subject only to applicable laws and
    regulations governing limitations on product liability.

    THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
    PART OF THIS FILE AT ALL TIMES.                       */

// #include "adf.h"
#include "../inc/lut_group2.h"
#include "aie_api/aie.hpp"

__attribute__((noinline)) void group2_copy(int16_t *__restrict px_in,
                                           int16_t *__restrict py_out) {

  v16bfloat16 *__restrict px = (v16bfloat16 *__restrict)px_in;
  v16bfloat16 *__restrict py = (v16bfloat16 *__restrict)py_out;

  v32bfloat16 a, b, c, d;

  // A A A X
  // B B B X  -> A A A B
  // C C C X  -> B B C C
  // D D D X  -> C D D D

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  for (int i = 0; i < 10; i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      a = set_v32bfloat16(0, *px++);
      b = set_v32bfloat16(0, *px++);
      *py++ = extract_v16bfloat16(insert(a, 3, extract_v4bfloat16(b, 0)), 0);
      c = set_v32bfloat16(0, *px++);
      *py++ = extract_v16bfloat16(
          insert(shift(b, undef_v32bfloat16(), (unsigned int)4), 1,
                 extract_v8bfloat16(c, 0)),
          0);
      d = set_v32bfloat16(0, *px++);
      *py++ = extract_v16bfloat16(
          insert(shift(undef_v32bfloat16(), d, (unsigned int)28), 0,
                 extract_v4bfloat16(c, 2)),
          0);
    }

  a = set_v32bfloat16(0, *px++);
  b = set_v32bfloat16(0, *px++);
  *py++ = extract_v16bfloat16(insert(a, 3, extract_v4bfloat16(b, 0)), 0);
  c = set_v32bfloat16(0, *px);
  *py =
      extract_v16bfloat16(insert(shift(b, undef_v32bfloat16(), (unsigned int)4),
                                 1, extract_v8bfloat16(c, 0)),
                          0);
}

__attribute__((noinline)) void group2_norm(int16_t *__restrict px_in,
                                           int16_t *__restrict lut,
                                           int16_t *__restrict py_out) {

  const int sz = 512;
  v16bfloat16 *__restrict px = (v16bfloat16 *__restrict)px_in;
  v16bfloat16 *__restrict py = (v16bfloat16 *__restrict)py_out;

  v32bfloat16 xa;
  v16accfloat acc0, acc1, acc2;

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  // 1/512
  v32bfloat16 coef = broadcast_to_v32bfloat16(-0.001953125);

  xa = set_v32bfloat16(0, *px++);
  xa = insert(xa, 1, *px++);

  acc0 = mul_elem_16_2(xa, coef);
  acc1 = mul_elem_16_2(xa, xa);

  for (int i = 0; i < (sz / 32 - 1); i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      xa = set_v32bfloat16(0, *px++);
      xa = insert(xa, 1, *px++);

      acc0 = mac_elem_16_2(xa, coef, acc0);
      acc1 = mac_elem_16_2(xa, xa, acc1);
    }

  px -= (sz >> 4);

  v32bfloat16 xb = concat(to_v16bfloat16(acc0), to_v16bfloat16(acc1));

  acc2 = mul_4x8_8x4(xb, broadcast_one_to_v32bfloat16());
  xb = shift(xb, undef_v32bfloat16(), (unsigned int)8);
  acc2 = mac_4x8_8x4(xb, broadcast_one_to_v32bfloat16(), acc2);

  v32bfloat16 xc = set_v32bfloat16(0, to_v16bfloat16(acc2));

  bfloat16 mean = ext_elem(xc, 0);

  bfloat16 xsqr = ext_elem(xc, 8);

  // compute variance
  v32bfloat16 xd = upd_elem(broadcast_zero_to_v32bfloat16(), 0, mean);
  xd = upd_elem(xd, 16, xsqr);
  coef = upd_elem(coef, 0, mean);

  int16_t var = ext_elem(
      set_v32int16(
          0, v16int16(to_v16bfloat16(mac_elem_16_2(
                 xd, coef, broadcast_to_v16accfloat(-9.999999974752427e-7))))),
      0);

  // extract low 8 bits to address look-up table
  int16_t lsb = var & 0x00ff;
  v32int16 u0 = upd_elem(broadcast_zero_to_v32int16(), 0, lut[lsb]);

  // (3.a) get exponent and do the inverse
  uint16 bb = (190 << 8) - (var & 0x7f00);

  // (3.b) convert the exponents into bfloat16 values and make sure the upper 16
  // values are 0
  v32int16 v0 =
      upd_elem(broadcast_zero_to_v32int16(), (int)0, (short)(bb >> 1));

  v16accfloat yy = mul_elem_16_2(v32bfloat16(u0), v32bfloat16(v0));

  bfloat16 scaling_fact = ext_elem(set_v32bfloat16(0, to_v16bfloat16(yy)), 0);

  v32bfloat16 ss = broadcast_to_v32bfloat16(scaling_fact);
  v32bfloat16 mm = broadcast_to_v32bfloat16(mean);

  // output scaling_fact * (x-mean)
  for (int i = 0; i < (sz >> 4); i++)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      mm = insert(mm, 0, *px++);
      *py++ = to_v16bfloat16(mul_elem_16_2(mm, ss));
    }
}

//---------------------------------
// 1x512 x 512xcol
//---------------------------------
__attribute__((noinline)) void
group2_mmul(int col_out, int16_t *__restrict px_in, int16_t *__restrict pc0_in,
            int16_t *__restrict pc1_in, int16_t *__restrict pb_in,
            int16_t *__restrict py_out) {

  const int din_len = 512;

  v16bfloat16 *__restrict px = (v16bfloat16 *__restrict)px_in;
  v16bfloat16 *__restrict pc0 = (v16bfloat16 *__restrict)pc0_in;
  v16bfloat16 *__restrict pc1 = (v16bfloat16 *__restrict)pc1_in;
  v4bfloat16 *__restrict pb = (v4bfloat16 *__restrict)pb_in;
  v8bfloat16 *__restrict py = (v8bfloat16 *__restrict)py_out;

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  for (int i = 0; i < (col_out / 8); i++)
    chess_prepare_for_pipelining chess_loop_range(3, ) {
      v32bfloat16 x = set_v32bfloat16(0, *px++); //(1)

      v16accfloat acc0 = mac_4x8_8x4(x, concat(*pc0++, *pc1++),
                                     ups_to_v16accfloat(extract_v16bfloat16(
                                         broadcast_to_v32bfloat16(*pb++), 0)));
      v16accfloat acc1 = mac_4x8_8x4(x, concat(*pc0++, *pc1++),
                                     ups_to_v16accfloat(extract_v16bfloat16(
                                         broadcast_to_v32bfloat16(*pb++), 0)));

      // 512/8 = 64
      // every loop has 2, so total 32 iterations
      for (int j = 0; j < (din_len / 8 / 2) - 1; j++)
        chess_flatten_loop {
          x = shift(x, undef_v32bfloat16(), (unsigned int)8);

          acc0 = mac_4x8_8x4(x, concat(*pc0++, *pc1++), acc0);
          acc1 = mac_4x8_8x4(x, concat(*pc0++, *pc1++), acc1);

          x = set_v32bfloat16(0, *px++); //(2)

          acc0 = mac_4x8_8x4(x, concat(*pc0++, *pc1++), acc0);
          acc1 = mac_4x8_8x4(x, concat(*pc0++, *pc1++), acc1);
        }

      x = shift(x, undef_v32bfloat16(), (unsigned int)8);
      px -= (din_len / 8 / 2);

      acc0 = mac_4x8_8x4(x, concat(*pc0++, *pc1++), acc0);
      v32bfloat16 ya = set_v32bfloat16(0, to_v16bfloat16(acc0));

      acc1 = mac_4x8_8x4(x, concat(*pc0++, *pc1++), acc1);
      v32bfloat16 yb = set_v32bfloat16(0, to_v16bfloat16(acc1));

      *py++ = extract_v8bfloat16(shuffle(ya, yb, T64_2x8_lo), 0);
    }
}

//--------------------------------------------
// log2
//--------------------------------------------
__attribute__((noinline)) void
softmax_exp_log2_224(int16_t *__restrict pd_in, int16_t *__restrict py_out) {

  v16int16 __aie_dm_resource_a *pd = (v16int16 __aie_dm_resource_a *)pd_in;
  v32int8 __aie_dm_resource_c *py = (v32int8 __aie_dm_resource_c *)py_out;

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  const v32acc32 offset = sups(broadcast_to_v32int16(127 * 128), 0);

  v32int16 a0, a1;
  v32acc32 b;

  // every time process 32 inputs
  // 224 / 32 = 7

  // load two data
  a0 = set_v32int16(0, *pd++);
  a0 = insert(a0, 1, *pd++);
  for (int i = 0; i < (224 / 32) >> 1; i++)
    chess_flatten_loop {
      // minus the offset
      b = sub(sups(a0, 0), offset);
      a1 = set_v32int16(0, *pd++);
      *py++ = ssrs(b, 3);
      a1 = insert(a1, 1, *pd++);

      b = sub(sups(a1, 0), offset);
      a0 = set_v32int16(0, *pd++);
      *py++ = ssrs(b, 3);
      a0 = insert(a0, 1, *pd++);
    }

  b = sub(sups(a0, 0), offset);
  *py++ = ssrs(b, 3);
}

//--------------------------------------------
// look up table is exp(-2^(x))
//--------------------------------------------
__attribute__((noinline)) void
softmax_exp_lkup_224(int16_t *__restrict pd_in, int16_t *__restrict lut_0,
                     int16_t *__restrict lut_1, int16_t *__restrict py_out) {

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  v32uint8 __aie_dm_resource_a *pd = (v32uint8 __aie_dm_resource_a *)pd_in;
  v16int16 __aie_dm_resource_b *py = (v16int16 __aie_dm_resource_b *)py_out;

  int8 __aie_dm_resource_c *lut0 = (int8 __aie_dm_resource_c *)lut_0;
  int8 __aie_dm_resource_d *lut1 = (int8 __aie_dm_resource_d *)lut_1;

  // load two data
  v32int32 b0, b1;

  v64int8 y1, y2, y3, y4;

  for (int i = 0; i < (224 / 32); i++)
    chess_flatten_loop {
      b0 = v32int32(sups(*pd++, 2));
      load_lut_2x_int8(lut0, lut1, extract_v16int32(b0, 0), y1, y2);
      *py++ =
          extract_v16int16(shuffle(v32int16(y1), v32int16(y2), T16_16x4_lo), 0);
      load_lut_2x_int8(lut0, lut1, extract_v16int32(b0, 1), y1, y2);
      *py++ =
          extract_v16int16(shuffle(v32int16(y1), v32int16(y2), T16_16x4_lo), 0);
    }
}

__attribute__((noinline)) bfloat16
softmax_findmax_224(int16_t *__restrict px_in) {

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  // find the max of 224 values
  v16bfloat16 *px = (v16bfloat16 *)px_in;
  v32bfloat16 a0, a1, b0, b1, c0;

  a0 = set_v32bfloat16(0, *px++);
  a0 = insert(a0, 1, *px++);
  a1 = set_v32bfloat16(0, *px++);
  a1 = insert(a1, 1, *px++);
  b0 = max(a0, a1);
  a0 = set_v32bfloat16(0, *px++);
  a0 = insert(a0, 1, *px++);
  a1 = set_v32bfloat16(0, *px++);
  a1 = insert(a1, 1, *px++);
  b1 = max(a0, a1);
  c0 = max(b0, b1);
  a0 = set_v32bfloat16(0, *px++);
  a0 = insert(a0, 1, *px++);
  a1 = set_v32bfloat16(0, *px++);
  a1 = insert(a1, 1, *px++);
  b0 = max(a0, a1);
  a0 = set_v32bfloat16(0, *px++);
  a0 = insert(a0, 1, *px);
  b1 = max(c0, a0);
  c0 = max(b0, b1);

  a0 = max(c0, shift(c0, undef_v32bfloat16(), (unsigned int)16));
  a1 = max(a0, shift(a0, undef_v32bfloat16(), (unsigned int)8));
  b0 = max(a1, shift(a1, undef_v32bfloat16(), (unsigned int)4));

  bfloat16 x0 = ext_elem(b0, 0);
  bfloat16 x1 = ext_elem(b0, 1);
  x0 = (x0 > x1) ? x0 : x1;
  x1 = ext_elem(b0, 2);
  x0 = (x0 > x1) ? x0 : x1;
  x1 = ext_elem(b0, 3);
  x0 = (x0 > x1) ? x0 : x1;

  return x0;
}

__attribute__((noinline)) void group2_softmax(int16_t *__restrict px_in,
                                              int16_t *__restrict lut0_in,
                                              int16_t *__restrict lut1_in,
                                              int16_t *__restrict lut_inv,
                                              int16_t *__restrict ptmp) {

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  v16bfloat16 *restrict px = (v16bfloat16 *restrict)px_in;
  v16bfloat16 *restrict py = (v16bfloat16 *restrict)ptmp;

  // force the values in the end to be very small
  {
    bfloat16 *pxx1 = (bfloat16 *)(px_in + 214);
    for (int i = 0; i < 10; i++)
      *pxx1++ = -1e25;
  }

  bfloat16 vmax = softmax_findmax_224(px_in);

  // Use max to minus the values to have everything positive before LOG2
  {
    v16accfloat acc0 = ups_to_v16accfloat(
        extract_v16bfloat16(broadcast_to_v32bfloat16(vmax), 0));

    for (int i = 0; i < (224 / 16); i++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        *py++ = to_v16bfloat16(sub(acc0, ups_to_v16accfloat(*px++)));
      }
    px -= (224 / 16);
    py -= (224 / 16);
  }

  // exp
  softmax_exp_log2_224(ptmp, px_in);
  softmax_exp_lkup_224(px_in, lut0_in, lut1_in, ptmp);

  // sum
  int16_t ssum;
  {
    v16accfloat sum = ups_to_v16accfloat(*py++);

    for (int i = 0; i < (224 / 16) - 1; i++)
      chess_flatten_loop { sum = add(sum, ups_to_v16accfloat(*py++)); }
    py -= (224 / 16);

    v32bfloat16 s0 = set_v32bfloat16(0, to_v16bfloat16(sum));

    sum = mul_4x8_8x4(s0, broadcast_one_to_v32bfloat16());
    s0 = shift(s0, undef_v32bfloat16(), (unsigned int)8);
    sum = mac_4x8_8x4(s0, broadcast_one_to_v32bfloat16(), sum);
    s0 = set_v32bfloat16(0, to_v16bfloat16(sum));
    ssum = ext_elem(v32int16(s0), 0);
  }

  // find the inverse
  bfloat16 scal;
  {
    // use low 7 bits to address the look-up table
    int addr = ssum & 0x007f;
    v32int16 u0 = upd_elem(broadcast_zero_to_v32int16(), 0, lut_inv[addr]);

    // get exponent and do the inverse
    int16_t c2 = 254 * 128 - (ssum & 0x7f80);

    // convert the exponents into bfloat16 values and make sure the upper 16
    // values are 0
    v32int16 v0 = upd_elem(broadcast_zero_to_v32int16(), 0, c2);

    // multiply exponent with the inserse of fraction part
    v16accfloat yy = mul_elem_16_2(v32bfloat16(u0), v32bfloat16(v0));

    scal = ext_elem(set_v32bfloat16(0, to_v16bfloat16(yy)), 0);
  }

  v32bfloat16 inv_vec = broadcast_to_v32bfloat16(scal);

  // scale the output
  {
    v8float *__restrict pyy = (v8float *__restrict)px_in;
    for (int i = 0; i < (224 / 16); i++)
      chess_flatten_loop {
        v16float yp = v16float(mul_elem_16_2(
            insert(broadcast_zero_to_v32bfloat16(), 0, *py++), inv_vec));

        *pyy++ = extract_v8float(yp, 0);
        *pyy++ = extract_v8float(yp, 1);
      }
  }
}

//--------------------------------
// Group 2
//--------------------------------
#define BA_C0a magika_2_a
#define BA_C1a magika_2_b
#define BA_C2a magika_2_c
#define BA_BUF2 magika_2_d

#define BA_C0b (BA_C0a + 16384)
#define BA_C1b (BA_C1a + 16384)
#define BA_C2b (BA_C2a + 16384)

#define BA_C3a (BA_BUF2 + 1024)
#define BA_SM0a (BA_C3a + 8192)
#define BA_SM1 (BA_SM0a + 512)
#define BA_NORM (BA_SM1 + 128)
#define BA_BIAS (BA_NORM + 256)
#define BA_BUF1 (BA_BIAS + 224)
#define BA_C3b (BA_BUF1 + 1024)
#define BA_SM0b (BA_C3b + 8192)

void group2(
    int16_t *__restrict xin,
    int16_t *__restrict magika_2_a, int16_t *__restrict magika_2_b,
    int16_t *__restrict magika_2_c, int16_t *__restrict magika_2_d) {

  //----------------------------------------
  // copy data in xin0 and xin1 to BUF1
  //----------------------------------------
  group2_copy(xin, BA_BUF1);

  //--------------------------------------
  // Normalization
  //--------------------------------------
  group2_norm(BA_BUF1, BA_NORM, BA_BUF2);

  //--------------------------------------
  // Matrix Multiplication
  //--------------------------------------
  // 64 channels each for first 64 outputs
  group2_mmul(64, BA_BUF2, BA_C0a, BA_C0b, BA_BIAS, BA_BUF1);
  group2_mmul(64, BA_BUF2, BA_C1a, BA_C1b, BA_BIAS + 64, BA_BUF1 + 64);
  group2_mmul(64, BA_BUF2, BA_C2a, BA_C2b, BA_BIAS + 128, BA_BUF1 + 128);

  // remaining 22 columns. packed to 32 columns
  group2_mmul(32, BA_BUF2, BA_C3a, BA_C3b, BA_BIAS + 192, BA_BUF1 + 192);

  //-----------------------------------------------------
  // Softmax
  //-----------------------------------------------------
  group2_softmax(BA_BUF1, BA_SM0a, BA_SM0b, BA_SM1, BA_BUF2);

  //-----------------------------------------------------
  // Output exactly 224 samples via streaming interface
  //------------------------------------------------------
  {
    int *px = (int *)BA_BUF1;
    for (int i = 0; i < 214; i++) {
      // put_ms(px[i]); // TODO yout int32 stream
      put_ms(px[i], (i == 213)); // TODO yout int32 stream
    }
  }
}

#undef BA_C0a
#undef BA_C1a
#undef BA_C2a
#undef BA_BUF2
#undef BA_C0b
#undef BA_C1b
#undef BA_C2b
#undef BA_C3a
#undef BA_SM0a
#undef BA_SM1
#undef BA_NORM
#undef BA_BIAS
#undef BA_BUF1
#undef BA_C3b
#undef BA_SM0b

extern "C" {

void group2_kernel(int16_t *__restrict xin, int16_t *__restrict magika_2_a,
                   int16_t *__restrict magika_2_b,
                   int16_t *__restrict magika_2_c,
                   int16_t *__restrict magika_2_d) {
  event0();
  group2(xin, magika_2_a, magika_2_b, magika_2_c, magika_2_d);
  event1();
}

} // extern "C"

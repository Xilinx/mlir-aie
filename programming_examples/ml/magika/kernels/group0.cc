/* Copyright (C) 2019-2022 Xilinx, Inc.
Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
SPDX-License-Identifier: LicenseRef-AMD-Proprietary */

// #include "adf.h"
// #include "../inc/lut_group0.h"
#include "aie_api/aie.hpp"

//--------------------------------
// Gather Layer
//--------------------------------

#ifdef GROUPA

void group0a(int16_t *xin, int16_t *yout, int16_t *magika_0a_cin, int32_t xid,
             int32_t cid) {

  int16_t *restrict px = xin;
  v8int16 *restrict pd = (v8int16 *restrict)magika_0a_cin;

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  int16_t *pxx = px + xid;

  v8int16 *restrict py = (v8int16 *restrict)yout;

  for (int k = 0; k < 512; k++)
    chess_prepare_for_pipelining chess_loop_range(64, ) chess_unroll_loop(8) {
      *py++ = pd[((*pxx) << 3) | cid];
      pxx += 4;
    }
}

#endif

#ifdef GROUPB
#include "../inc/funcs_layernorm.h"

void group0b(int16_t *xin, int16_t *yout, int16_t *magika_0b_cin_a,
             int16_t *magika_0b_cin_b) {

  int16_t *restrict px = xin;
  int16_t *restrict pout = yout;

  int16_t *BA_LUT0 = magika_0b_cin_a;
  int16_t *BA_LUT1 = magika_0b_cin_b;

  int16_t *BA_G = (BA_LUT0 + 512);
  int16_t *BA_B = (BA_LUT1 + 512);

  int16_t *BA_BUF = (BA_G + 4096);

  aie::set_saturation(aie::saturation_mode::saturate);
  aie::set_rounding(aie::rounding_mode::symmetric_inf);

  v8bfloat16 mean, var;

  layernorm_sum_row_512x8(px, mean, var, pout);

  // compute the denominator
  var = layernorm_denom(var, BA_LUT0, BA_LUT1);

  // normalization
  layernorm_norm_512x8(pout, mean, var, BA_BUF);

  // rescale to target
  layernorm_rescale_512x8(BA_BUF, BA_G, BA_B, pout);
}

#endif

extern "C" {

#ifdef GROUPA
void group0a_kernel(int16_t *xin, int16_t *yout, int16_t *magika_0a_cin,
                    int32_t xid, int32_t cid) {
  event0();
  group0a(xin, yout, magika_0a_cin, xid, cid);
  event1();
}
#endif

#ifdef GROUPB
void group0b_kernel(int16_t *xin, int16_t *yout, int16_t *magika_0b_cin_a,
                    int16_t *magika_0b_cin_b) {
  event0();
  group0b(xin, yout, magika_0b_cin_a, magika_0b_cin_b);
  event1();
}
#endif

} // extern "C"

/*  (c) Copyright 2019-2022 Xilinx, Inc. All rights reserved.
    (c) Copyright 2022-2024 Advanced Micro Devices, Inc. All rights reserved.

    This file contains confidential and proprietary information
    of Xilinx, Inc. and is protected under U.S. and
    international copyright and other intellectual property
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
    injury, or severe property or environmental damage
    (individually and collectively, "Critical
    Applications"). Customer assumes the sole risk and
    liability of any use of Xilinx products in Critical
    Applications, subject only to applicable laws and
    regulations governing limitations on product liability.

    THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
    PART OF THIS FILE AT ALL TIMES.                       */

#pragma once
// #include "adf.h"
#include "aie_api/aie.hpp"

//----------------------------------
// RESET Output Buffer to Bias
//----------------------------------
void conv1d_12x8x512_init(int16 *pb_in, int16 *py_out) {

  // load the bias
  v4bfloat16 *__restrict pb = (v4bfloat16 *__restrict)pb_in;

  // pointer to output
  v16bfloat16 *__restrict py = (v16bfloat16 *__restrict)py_out;

  // every time write out 1 ogrp x 4 och x 4 pixels
  for (int ogrp = 0; ogrp < 3; ogrp++) {
    v16bfloat16 bb = extract_v16bfloat16(broadcast_to_v32bfloat16(*pb++), 0);

    for (int i = 0; i < (508 / 4); i++)
      chess_prepare_for_pipelining chess_loop_range(8, ) { *py++ = bb; }
  }
}

//--------------------------------------------
// Accumulate. 8 input channels every time
//--------------------------------------------
void conv1d_12x8x512x5(int16 *pd_in, int16 *pc0_in, int16 *pc1_in, int16 *py) {

  v16bfloat16 __aie_dm_resource_b *__restrict pc0 =
      (v16bfloat16 __aie_dm_resource_b *__restrict)pc0_in;
  v16bfloat16 __aie_dm_resource_c *__restrict pc1 =
      (v16bfloat16 __aie_dm_resource_c *__restrict)pc1_in;

  v32bfloat16 c0, c1, c2, c3, c4;
  v32bfloat16 a0, a1;
  v16accfloat vy0, vy1;

  // Y_o = Y_i + X_512x8 * C1_8x12 + SHFT(X_512x8) * C2_8x12 + ... SHFT(X_512x8)
  // * C5_8x12

  // first 4 channels
  {
    // preload all A values
    c0 = concat(*pc0++, *pc1++);
    c1 = concat(*pc0++, *pc1++);
    c2 = concat(*pc0++, *pc1++);
    0 c3 = concat(*pc0++, *pc1++);
    c4 = concat(*pc0++, *pc1++);

    v16bfloat16 __aie_dm_resource_a *__restrict pd =
        (v16bfloat16 __aie_dm_resource_a *__restrict)pd_in;
    v16bfloat16 __aie_dm_resource_d *__restrict pi =
        (v16bfloat16 __aie_dm_resource_d *__restrict)py;
    v16bfloat16 __aie_dm_resource_d *__restrict po = chess_copy(pi);

    a0 = set_v32bfloat16(0, *pd++); // 0, 1
    a0 = insert(a0, 1, *pd++);      // 0, 1, 2, 3

    for (int i = 0; i < 504 / 4 / 2; i++)
      chess_prepare_for_pipelining chess_loop_range(63, 63)
          chess_unroll_loop(3) {
        vy0 = mac_4x8_8x4(a0, c0, ups(*pi++));
        a1 = set_v32bfloat16(0, *pd++);
        vy0 = mac_4x8_8x4(shift(a0, a1, 8), c1, vy0);
        a1 = insert(a1, 1, *pd++);
        vy0 = mac_4x8_8x4(shift(a0, a1, 16), c2, vy0);
        vy0 = mac_4x8_8x4(shift(a0, a1, 24), c3, vy0);
        vy0 = mac_4x8_8x4(a1, c4, vy0);
        *po++ = to_v16bfloat16(vy0);

        vy1 = mac_4x8_8x4(a1, c0, ups(*pi++));
        a0 = set_v32bfloat16(0, *pd++);
        vy1 = mac_4x8_8x4(shift(a1, a0, 8), c1, vy1);
        a0 = insert(a0, 1, *pd++);
        vy1 = mac_4x8_8x4(shift(a1, a0, 16), c2, vy1);
        vy1 = mac_4x8_8x4(shift(a1, a0, 24), c3, vy1);
        vy1 = mac_4x8_8x4(a0, c4, vy1);
        *po++ = to_v16bfloat16(vy1);
      }

    vy0 = mac_4x8_8x4(a0, c0, ups(*pi++));
    a1 = set_v32bfloat16(0, *pd++);
    vy0 = mac_4x8_8x4(shift(a0, a1, 8), c1, vy0);
    a1 = insert(a1, 1, *pd++);
    vy0 = mac_4x8_8x4(shift(a0, a1, 16), c2, vy0);
    vy0 = mac_4x8_8x4(shift(a0, a1, 24), c3, vy0);
    vy0 = mac_4x8_8x4(a1, c4, vy0);
    *po = to_v16bfloat16(vy0);
  }

  // next 8 channels
  {

    v32bfloat16 a2, c5, c6, c7, c8, c9;
    v32bfloat16 ax, ay;

    v16bfloat16 __aie_dm_resource_a *__restrict pd =
        (v16bfloat16 __aie_dm_resource_a *__restrict)pd_in;

    v16bfloat16 __aie_dm_resource_d *__restrict pi0 =
        (v16bfloat16 __aie_dm_resource_d *__restrict)(py + 4 * 508);
    v16bfloat16 __aie_dm_resource_d *__restrict po0 = chess_copy(pi0);

    v16bfloat16 __aie_dm_resource_d *__restrict pi1 = chess_copy(pi0 + 508 / 4);
    v16bfloat16 __aie_dm_resource_d *__restrict po1 = chess_copy(pi1);

    // preload all A values
    c0 = concat(*pc0++, *pc1++);
    c1 = concat(*pc0++, *pc1++);
    c2 = concat(*pc0++, *pc1++);
    c3 = concat(*pc0++, *pc1++);
    c4 = concat(*pc0++, *pc1++);
    c5 = concat(*pc0++, *pc1++);
    c6 = concat(*pc0++, *pc1++);
    c7 = concat(*pc0++, *pc1++);
    c8 = concat(*pc0++, *pc1++);
    c9 = concat(*pc0++, *pc1++);

    a0 = set_v32bfloat16(0, *pd++); // 0, 1
    a0 = insert(a0, 1, *pd++);      // 0, 1, 2, 3

    for (int i = 0; i < 504 / 4 / 2; i++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        vy0 = mac_4x8_8x4(a0, c0, ups(*pi0++));
        a1 = set_v32bfloat16(0, *pd++); // 4, 5
        vy1 = mac_4x8_8x4(a0, c5, ups(*pi1++));
        ax = shift(a0, a1, 8);

        vy0 = mac_4x8_8x4(ax, c1, vy0);
        ay = shift(a0, a1, 16);
        vy1 = mac_4x8_8x4(ax, c6, vy1);
        a1 = insert(a1, 1, *pd++);

        vy0 = mac_4x8_8x4(ay, c2, vy0);
        vy1 = mac_4x8_8x4(ay, c7, vy1);
        ax = shift(a0, a1, 24);

        vy0 = mac_4x8_8x4(ax, c3, vy0);
        vy1 = mac_4x8_8x4(ax, c8, vy1);

        vy0 = mac_4x8_8x4(a1, c4, vy0);
        *po0++ = to_v16bfloat16(vy0);
        vy1 = mac_4x8_8x4(a1, c9, vy1);
        *po1++ = to_v16bfloat16(vy1);

        vy0 = mac_4x8_8x4(a1, c0, ups(*pi0++));
        a0 = set_v32bfloat16(0, *pd++); // 4, 5
        vy1 = mac_4x8_8x4(a1, c5, ups(*pi1++));
        ay = shift(a1, a0, 8);

        vy0 = mac_4x8_8x4(ay, c1, vy0);
        ax = shift(a1, a0, 16);
        vy1 = mac_4x8_8x4(ay, c6, vy1);
        a0 = insert(a0, 1, *pd++);

        vy0 = mac_4x8_8x4(ax, c2, vy0);
        vy1 = mac_4x8_8x4(ax, c7, vy1);
        ay = shift(a1, a0, 24);

        vy0 = mac_4x8_8x4(ay, c3, vy0);
        vy1 = mac_4x8_8x4(ay, c8, vy1);

        vy0 = mac_4x8_8x4(a0, c4, vy0);
        *po0++ = to_v16bfloat16(vy0);
        vy1 = mac_4x8_8x4(a0, c9, vy1);
        *po1++ = to_v16bfloat16(vy1);
      }

    vy0 = mac_4x8_8x4(a0, c0, ups(*pi0++));
    a1 = set_v32bfloat16(0, *pd++); // 4, 5
    vy1 = mac_4x8_8x4(a0, c5, ups(*pi1++));
    ax = shift(a0, a1, 8);

    vy0 = mac_4x8_8x4(ax, c1, vy0);
    ay = shift(a0, a1, 16);
    vy1 = mac_4x8_8x4(ax, c6, vy1);
    a1 = insert(a1, 1, *pd++);

    vy0 = mac_4x8_8x4(ay, c2, vy0);
    vy1 = mac_4x8_8x4(ay, c7, vy1);
    ax = shift(a0, a1, 24);

    vy0 = mac_4x8_8x4(ax, c3, vy0);
    vy1 = mac_4x8_8x4(ax, c8, vy1);

    vy0 = mac_4x8_8x4(a1, c4, vy0);
    *po0 = to_v16bfloat16(vy0);
    vy1 = mac_4x8_8x4(a1, c9, vy1);
    *po1 = to_v16bfloat16(vy1);
  }
}

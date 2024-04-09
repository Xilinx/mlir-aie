//===- hdiff.cc -------------------------------------------------*- C++ -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
//
//===----------------------------------------------------------------------===//

#include "hdiff.h"
#include "./include.h"
#define kernel_load 14

void vec_hdiff(int32_t *restrict row0, int32_t *restrict row1,
               int32_t *restrict row2, int32_t *restrict row3,
               int32_t *restrict row4, int32_t *restrict out) {

  alignas(32) int32_t weights[8] = {-4, -4, -4, -4, -4, -4, -4, -4};
  alignas(32) int32_t weights1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  alignas(32) int32_t weights_rest[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  alignas(32) int32_t flux_out[8] = {-7, -7, -7, -7, -7, -7, -7, -7};

  v8int32 coeffs = *(v8int32 *)weights;           //  8 x int32 = 256b W vector
  v8int32 coeffs1 = *(v8int32 *)weights1;         //  8 x int32 = 256b W vector
  v8int32 coeffs_rest = *(v8int32 *)weights_rest; //  8 x int32 = 256b W vector
  v8int32 flux_out_coeff = *(v8int32 *)flux_out;

  v8int32 *ptr_out = (v8int32 *)out;
  v8int32 *restrict row0_ptr = (v8int32 *)row0;
  v8int32 *restrict row1_ptr = (v8int32 *)row1;
  v8int32 *restrict row2_ptr = (v8int32 *)row2;
  v8int32 *restrict row3_ptr = (v8int32 *)row3;
  v8int32 *restrict row4_ptr = (v8int32 *)row4;
  v8int32 *restrict r1;

  v16int32 data_buf1 = null_v16int32();
  v16int32 data_buf2 = null_v16int32();
  ;

  v8acc80 acc_0 = null_v8acc80();
  v8acc80 acc_1 = null_v8acc80();

  v8int32 lap_ij = null_v8int32(); //  8 x int32 = 256b W vector
  v8int32 lap_0 = null_v8int32();  //  8 x int32 = 256b W vector

  data_buf1 = upd_w(data_buf1, 0, *row3_ptr++);
  data_buf1 = upd_w(data_buf1, 1, *row3_ptr);
  data_buf2 = upd_w(data_buf2, 0, *row1_ptr++);
  data_buf2 = upd_w(data_buf2, 1, *row1_ptr);

  for (unsigned i = 0; i < COL / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      v16int32 flux_sub;

      acc_0 = lmul8(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); // c
      acc_1 = lmul8(data_buf2, 1, 0x76543210, coeffs_rest, 0, 0x00000000); // b

      acc_0 = lmac8(acc_0, data_buf1, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); // c,k
      acc_1 = lmac8(acc_1, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); // b,j

      // r2 = ptr_in+2 * COL/8+i ;
      // r1=row2_ptr+i;
      row2_ptr = ((v8int32 *)(row2)) + i;
      data_buf2 = upd_w(data_buf2, 0, *(row2_ptr)++);
      data_buf2 = upd_w(data_buf2, 1, *(row2_ptr));

      acc_0 = lmac8(acc_0, data_buf2, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); // c,k,f
      acc_0 = lmsc8(acc_0, data_buf2, 2, 0x76543210, coeffs, 0,
                    0x00000000); // c,k,f,4*g
      acc_0 = lmac8(acc_0, data_buf2, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); // c,k,f,4*g,h

      lap_ij = srs(acc_0, 0); // store lap_ij

      acc_1 = lmac8(acc_1, data_buf2, 0, 0x76543210, coeffs_rest, 0,
                    0x00000000); // b,j,e
      acc_1 = lmsc8(acc_1, data_buf2, 1, 0x76543210, coeffs, 0,
                    0x00000000); // b,j,e,4*f
      acc_1 = lmac8(acc_1, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); // b,j,e,4*f,g

      // lap_ijm
      lap_0 = srs(acc_1, 0);

      // Calculate  lap_ij - lap_ijm
      flux_sub =
          sub16(concat(lap_ij, undef_v8int32()), 0, 0x76543210, 0xFEDCBA98,
                concat(lap_0, undef_v8int32()), 0, 0x76543210, 0xFEDCBA98);

      //
      acc_1 = lmul8(data_buf2, 2, 0x76543210, ext_w(flux_sub, 0), 0,
                    0x00000000); // (lap_ij - lap_ijm)*g
      acc_1 = lmsc8(acc_1, data_buf2, 1, 0x76543210, ext_w(flux_sub, 0), 0,
                    0x00000000); // (lap_ij - lap_ijm)*g - (lap_ij - lap_ijm)*f

      // compare > 0
      unsigned int flx_compare_imj =
          gt16(concat(srs(acc_1, 0), undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);

      acc_0 = lmul8(data_buf1, 3, 0x76543210, coeffs_rest, 0, 0x00000000); // l
      acc_0 = lmsc8(acc_0, data_buf2, 3, 0x76543210, coeffs, 0,
                    0x00000000); // l, 4*h

      // Calculate final fly_ijm
      v16int32 out_flx_inter1 =
          select16(flx_compare_imj, flux_sub, null_v16int32());

      // r1 = ptr_in+1 * COL/8+i ;
      row1_ptr = ((v8int32 *)(row1)) + i;
      data_buf1 = upd_w(data_buf1, 0, *(row1_ptr)++);
      data_buf1 = upd_w(data_buf1, 1, *(row1_ptr));

      acc_0 = lmac8(acc_0, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); // l, 4*h, g
      acc_0 = lmac8(acc_0, data_buf2, 4, 0x76543210, coeffs_rest, 0,
                    0x00000000); // l, 4*h, g, i
      v16int32 flx_out1 =
          add16(null_v16int32(), out_flx_inter1); // still fly_ijm

      acc_0 = lmac8(acc_0, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); // l, 4*h, g, i, d

      // Calculates lap_ijp
      lap_0 = srs(acc_0, 0);

      // Calculates lap_ijp - lap_ij
      flux_sub =
          sub16(concat(lap_0, undef_v8int32()), 0, 0x76543210, 0xFEDCBA98,
                concat(lap_ij, undef_v8int32()), 0, 0x76543210, 0xFEDCBA98);

      acc_0 = lmul8(data_buf2, 3, 0x76543210, ext_w(flux_sub, 0), 0,
                    0x00000000); // (lap_ijp - lap_ij) * h
      acc_0 = lmsc8(
          acc_0, data_buf2, 2, 0x76543210, ext_w(flux_sub, 0), 0,
          0x00000000); //  (lap_ijp - lap_ij) * h  - (lap_ijp - lap_ij) * g

      // Calculates final fly_ij (comparison > 0)
      unsigned int flx_compare_ij =
          gt16(concat(srs(acc_0, 0), undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);
      v16int32 out_flx_inter2 =
          select16(flx_compare_ij, flux_sub, null_v16int32());

      // add fly_ij - fly_ijm
      v16int32 flx_out2 = sub16(out_flx_inter2, flx_out1);

      //***********************************************************************STARTING
      //X
      //FLUX*****************************************************************************************************************************************************

      acc_1 = lmul8(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); // g
      acc_0 = lmul8(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); // g

      row0_ptr = ((v8int32 *)(row0)) + i;
      data_buf2 = upd_w(data_buf2, 0, *(row0_ptr)++);
      data_buf2 = upd_w(data_buf2, 1, *(row0_ptr));

      acc_1 = lmsc8(acc_1, data_buf1, 2, 0x76543210, coeffs, 0,
                    0x00000000); // g, 4*c
      acc_1 = lmac8(acc_1, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, 4*c, b
      acc_1 = lmac8(acc_1, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, 4*c, b, a

      row4_ptr = ((v8int32 *)(row4)) + i;
      data_buf2 = upd_w(data_buf2, 0, *(row4_ptr)++);
      data_buf2 = upd_w(data_buf2, 1, *(row4_ptr));

      acc_1 = lmac8(acc_1, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, 4*c, b, a, d
      acc_0 = lmac8(acc_0, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, m

      row2_ptr = ((v8int32 *)(row2)) + i;
      data_buf2 = upd_w(data_buf2, 0, *(row2_ptr)++);
      data_buf2 = upd_w(data_buf2, 1, *(row2_ptr));

      // Calculates lap_imj
      lap_0 = srs(acc_1, 0);

      // flx_imj = lap_ij - lap_imj
      flux_sub =
          sub16(concat(lap_ij, undef_v8int32()), 0, 0x76543210, 0xFEDCBA98,
                concat(lap_0, undef_v8int32()), 0, 0x76543210, 0xFEDCBA98);

      acc_1 = lmul8(data_buf2, 2, 0x76543210, ext_w(flux_sub, 0), 0,
                    0x00000000); //   (lap_ij - lap_imj) * g
      acc_1 = lmsc8(
          acc_1, data_buf1, 2, 0x76543210, ext_w(flux_sub, 0), 0,
          0x00000000); //    (lap_ij - lap_imj) * g  *  (lap_ij - lap_imj) * c

      // Calculates final flx_imj (comparison > 0)
      unsigned int fly_compare_ijm =
          gt16(concat(srs(acc_1, 0), undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);
      v16int32 out_flx_inter3 =
          select16(fly_compare_ijm, flux_sub, null_v16int32());

      row3_ptr = ((v8int32 *)(row3)) + i;
      data_buf1 = upd_w(data_buf1, 0, *(row3_ptr)++);
      data_buf1 = upd_w(data_buf1, 1, *(row3_ptr));

      acc_0 = lmsc8(acc_0, data_buf1, 2, 0x76543210, coeffs, 0,
                    0x00000000); // g, m , k * 4

      v16int32 flx_out3 =
          sub16(flx_out2, out_flx_inter3); // adds fly_ij - fly_ijm - flx_imj

      acc_0 = lmac8(acc_0, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, m , k * 4, j
      acc_0 = lmac8(acc_0, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, m , k * 4, j, l

      //  flx_ij = lap_ipj - lap_ij
      flux_sub = sub16(concat(srs(acc_0, 0), undef_v8int32()), 0, 0x76543210,
                       0xFEDCBA98, concat(lap_ij, undef_v8int32()), 0,
                       0x76543210, 0xFEDCBA98);

      // below reuisng acc_1 for flux calculation //CHANGED FROM 3 TO 2
      acc_1 = lmul8(data_buf1, 2, 0x76543210, ext_w(flux_sub, 0), 0,
                    0x00000000); //  (lap_ipj - lap_ij) * k

      // LOAD DATA FOR NEXT ITERATION
      row3_ptr = ((v8int32 *)(row3)) + i + 1;
      data_buf1 = upd_w(data_buf1, 0, *(row3_ptr)++);
      data_buf1 = upd_w(data_buf1, 1, *(row3_ptr));

      acc_1 =
          lmsc8(acc_1, data_buf2, 2, 0x76543210, ext_w(flux_sub, 0), 0,
                0x00000000); //  (lap_ipj - lap_ij) * k - (lap_ipj - lap_ij) * g

      // final flx_ij (comparison > 0 )
      unsigned int fly_compare_ij =
          gt16(concat(srs(acc_1, 0), undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);
      v16int32 out_flx_inter4 =
          select16(fly_compare_ij, flux_sub, null_v16int32());

      v16int32 flx_out4 = add16(
          flx_out3, out_flx_inter4); // adds fly_ij - fly_ijm - flx_imj + flx_ij

      v8acc80 final_output = lmul8(flx_out4, 0, 0x76543210, flux_out_coeff, 0,
                                   0x00000000); // Multiply by -7s
      final_output = lmac8(final_output, data_buf2, 2, 0x76543210,
                           concat(coeffs1, undef_v8int32()), 0, 0x76543210);

      // LOAD DATA FOR NEXT ITERATION
      row1_ptr = ((v8int32 *)(row1)) + i + 1;
      data_buf2 = upd_w(data_buf2, 0, *(row1_ptr)++);
      data_buf2 = upd_w(data_buf2, 1, *(row1_ptr));

      *ptr_out++ = srs(final_output, 0);
    }
}

//===- hdiff_fp32.cc --------------------------------------------*- C++ -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
//
//===----------------------------------------------------------------------===//

#include "./include.h"
#include "hdiff.h"
#define kernel_load 14

void vec_hdiff_fp32(float *restrict row0, float *restrict row1,
                    float *restrict row2, float *restrict row3,
                    float *restrict row4, float *restrict out) {

  alignas(32) float weights[8] = {-4, -4, -4, -4, -4, -4, -4, -4};
  alignas(32) float weights1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  alignas(32) float weights_rest[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  alignas(32) float flux_out[8] = {-7, -7, -7, -7, -7, -7, -7, -7};

  v8float coeffs = *(v8float *)weights;           //  8 x int32 = 256b W vector
  v8float coeffs1 = *(v8float *)weights1;         //  8 x int32 = 256b W vector
  v8float coeffs_rest = *(v8float *)weights_rest; //  8 x int32 = 256b W vector
  v8float flux_out_coeff = *(v8float *)flux_out;

  v8float *ptr_out = (v8float *)out;
  v8float *restrict row0_ptr = (v8float *)row0;
  v8float *restrict row1_ptr = (v8float *)row1;
  v8float *restrict row2_ptr = (v8float *)row2;
  v8float *restrict row3_ptr = (v8float *)row3;
  v8float *restrict row4_ptr = (v8float *)row4;
  v8float *restrict r1;

  v16float data_buf1 = null_v16float();
  v16float data_buf2 = null_v16float();

  v8float acc_0 = null_v8float();
  v8float acc_1 = null_v8float();

  v8float lap_ij = null_v8float(); //  8 x int32 = 256b W vector
  v8float lap_0 = null_v8float();  //  8 x int32 = 256b W vector

  data_buf1 = upd_w(data_buf1, 0, *row3_ptr++);
  data_buf1 = upd_w(data_buf1, 1, *row3_ptr);
  data_buf2 = upd_w(data_buf2, 0, *row1_ptr++);
  data_buf2 = upd_w(data_buf2, 1, *row1_ptr);

  for (unsigned i = 0; i < COL / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {

      v8float flux_sub;

      // buf_2=R1, and buf_1=R3
      /////////
      ///**************************LAP_ij**************************************
      lap_ij =
          fpmul(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); ///  //c
      /////////
      ///**************************LAP_ijm**************************************
      acc_1 = fpmul(data_buf2, 1, 0x76543210, coeffs_rest, 0, 0x00000000); // b

      /////////
      ///**************************LAP_ij**************************************
      lap_ij = fpmac(lap_ij, data_buf1, 2, 0x76543210, coeffs_rest, 0,
                     0x00000000); ///  //c,k
      /////////
      ///**************************LAP_imj**************************************
      acc_1 = fpmac(acc_1, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  //b,j

      row2_ptr = ((v8float *)(row2)) + i;

      data_buf2 = upd_w(data_buf2, 0, *row2_ptr++);
      data_buf2 = upd_w(data_buf2, 1, *row2_ptr);

      /////////
      ///**************************LAP_ij**************************************
      lap_ij = fpmac(lap_ij, data_buf2, 1, 0x76543210, coeffs_rest, 0,
                     0x00000000); ///  //c,k,f
      lap_ij = fpmsc(lap_ij, data_buf2, 2, 0x76543210, coeffs, 0,
                     0x00000000); ///  //c,k,f,4*g
      lap_ij = fpmac(
          lap_ij, data_buf2, 3, 0x76543210, coeffs_rest, 0,
          0x00000000); ///  c,k,f,4*g,h
                       /////////
                       ///**************************LAP_ijm**************************************
      acc_1 = fpmac(acc_1, data_buf2, 0, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  //b,j,e
      acc_1 = fpmsc(acc_1, data_buf2, 1, 0x76543210, coeffs, 0,
                    0x00000000); ///         //b,j,e,4*f
      acc_1 = fpmac(acc_1, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///     //b,j,e,4*f,g

      // Calculate  lap_ij - lap_ijm
      flux_sub = fpsub(lap_ij, concat(acc_1, undef_v8float()), 0,
                       0x76543210); // flx_imj = lap_ij - lap_ijm;

      //*****************FLUX //below reuisng acc_1 for flux calculation
      acc_1 = fpmul(data_buf2, 2, 0x76543210, flux_sub, 0,
                    0x00000000); ///  (lap_ij - lap_ijm)*g
      acc_1 = fpmsc(acc_1, data_buf2, 1, 0x76543210, flux_sub, 0,
                    0x00000000); // (lap_ij - lap_ijm)*g - (lap_ij - lap_ijm)*f

      // compare > 0
      unsigned int flx_compare_imj =
          fpge(acc_1, null_v16float(), 0,
               0x76543210); /// flx_ijm * (test_in[d][c][r] -
                            /// test_in[d][c][r-1]) > 0 ? 0 :

      // ///////// **************************lap_ipj
      // flx_ij**************************************
      acc_0 = fpmul(data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); /// // l ; R1 is already loaded
      // ///////// **************************lap_ipj
      // flx_ij**************************************
      acc_0 = fpmsc(acc_0, data_buf2, 3, 0x76543210, coeffs, 0,
                    0x00000000); ///   // l, 4*h

      // Calculate final fly_ijm
      v16float out_flx_inter1 = fpselect16(
          flx_compare_imj, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      row1_ptr = ((v8float *)(row1)) + i;

      data_buf1 = upd_w(data_buf1, 0, *row1_ptr++);
      data_buf1 = upd_w(data_buf1, 1, *row1_ptr);

      acc_0 = fpmac(acc_0, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  lap_ijp // l, 4*h, g
      acc_0 = fpmac(acc_0, data_buf2, 4, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  l, 4*h, g, i
      v8float flx_out1 =
          fpadd(null_v8float(), out_flx_inter1, 0, 0x76543210); // still fly_ijm

      acc_0 = fpmac(acc_0, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  // l, 4*h, g, i, d

      // Calculates lap_ijp - lap_ij
      flux_sub = fpsub(acc_0, concat(lap_ij, undef_v8float()), 0, 0x76543210);

      acc_1 = fpmul(data_buf2, 3, 0x76543210, flux_sub, 0,
                    0x00000000); // (lap_ijp - lap_ij) * h
      acc_1 = fpmsc(
          acc_1, data_buf2, 2, 0x76543210, flux_sub, 0,
          0x00000000); //  (lap_ijp - lap_ij) * h  - (lap_ijp - lap_ij) * g

      // Calculates final fly_ij (comparison > 0)
      unsigned int flx_compare_ij = fpge(acc_1, null_v16float(), 0, 0x76543210);
      v16float out_flx_inter2 = fpselect16(
          flx_compare_ij, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      // fly_ijm -fly_ij
      v8float flx_out2 = fpsub(flx_out1, out_flx_inter2, 0, 0x76543210);
      // NOTE: fbsub does not support v16-v8 so instead of calculating fly_ij -
      // fly_ijm - flx_imj + flx_ij
      //               we calculate -(-fly_ij + fly_ijm + flx_imj - flx_ij)
      //  Therefore, flux coeff is positive as well.

      // //***********************************************************************STARTING
      // X
      // FLUX*****************************************************************************************************************************************************
      //            //////// **************************LAP_imj sincer2=R2 are
      //            already loaded********
      acc_1 = fpmul(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); // g

      //////// **************************LAP_ipj for fly_ij sincer2=R2 are
      /// already loaded********
      acc_0 = fpmul(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); // g

      row0_ptr = ((v8float *)(row0)) + i;
      // r2 = ptr_in + 0*COL/8 + i + aor*COL/8;  // load for LAP_ijm for fly_ijm
      data_buf2 = upd_w(data_buf2, 0, *row0_ptr++);
      data_buf2 = upd_w(data_buf2, 1, *row0_ptr);

      // ///////// **************************LAP_imj since R1=buff 1
      // ************************************

      acc_1 = fpmsc(acc_1, data_buf1, 2, 0x76543210, coeffs, 0,
                    0x00000000); // g, 4*c
      acc_1 = fpmac(acc_1, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, 4*c, b
      acc_1 = fpmac(acc_1, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); /// // g, 4*c, b, a

      row4_ptr = ((v8float *)(row4)) + i;
      // load for LAP_ijm for fly_ijm
      data_buf2 = upd_w(data_buf2, 0, *row4_ptr++);
      data_buf2 = upd_w(data_buf2, 1, *row4_ptr);
      //////// **************************LAP_ipj for fly_ij since r2=R4********
      acc_1 = fpmac(acc_1, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, 4*c, b, a, d
      acc_0 = fpmac(acc_0, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///   // g, m

      row2_ptr = ((v8float *)(row2)) + i;
      // load for LAP_ijm for fly_ijm
      data_buf2 = upd_w(data_buf2, 0, *row2_ptr++);
      data_buf2 = upd_w(data_buf2, 1, *row2_ptr);

      // flx_imj = lap_ij - lap_imj
      flux_sub = fpsub(lap_ij, concat(acc_1, undef_v8float()), 0, 0x76543210);

      acc_1 = fpmul(data_buf2, 2, 0x76543210, flux_sub, 0,
                    0x00000000); ///   (lap_ij - lap_imj) * g
      acc_1 = fpmsc(
          acc_1, data_buf1, 2, 0x76543210, flux_sub, 0,
          0x00000000); ///    (lap_ij - lap_imj) * g  *  (lap_ij - lap_imj) * c

      // Calculates final flx_imj (comparison > 0)
      unsigned int fly_compare_ijm =
          fpge(acc_1, null_v16float(), 0, 0x76543210);
      v16float out_flx_inter3 = fpselect16(
          fly_compare_ijm, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      row3_ptr = ((v8float *)(row3)) + i;
      // load for LAP_ijp for fly_ij
      // data_buf1=*r1++;
      data_buf1 = upd_w(data_buf1, 0, *row3_ptr++);
      data_buf1 = upd_w(data_buf1, 1, *row3_ptr);
      ////// **************************since r1=R1********
      acc_0 = fpmsc(acc_0, data_buf1, 2, 0x76543210, coeffs, 0,
                    0x00000000); ///  //g, m , k * 4

      v8float flx_out3 = fpadd(flx_out2, out_flx_inter3, 0,
                               0x76543210); // adds fly_ijm -fly_ij + flx_imj

      acc_0 = fpmac(acc_0, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  //g, m , k * 4, j
      acc_0 = fpmac(acc_0, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///   //g, m , k * 4, j, l
      // lap_0=srs(acc_0,0); //LAP_ijp
      ///////// **************************fly_ij
      ///*************************************
      //  flx_ij = lap_ipj - lap_ij
      flux_sub = fpsub(acc_0, concat(lap_ij, undef_v8float()), 0, 0x76543210);

      // below reuisng acc_1 for flux calculation //CHANGED FROM 3 TO 2
      acc_1 = fpmul(data_buf1, 2, 0x76543210, flux_sub, 0,
                    0x00000000); //  (lap_ipj - lap_ij) * k

      // LOAD DATA FOR NEXT ITERATION
      row3_ptr = ((v8float *)(row3)) + i + 1;

      // data_buf1=*r1++;
      data_buf1 = upd_w(data_buf1, 0, *row3_ptr++);
      data_buf1 = upd_w(data_buf1, 1, *row3_ptr);

      acc_1 =
          fpmsc(acc_1, data_buf2, 2, 0x76543210, flux_sub, 0,
                0x00000000); //  (lap_ipj - lap_ij) * k - (lap_ipj - lap_ij) * g

      // final flx_ij (comparison > 0 )
      unsigned int fly_compare_ij = fpge(acc_1, null_v16float(), 0, 0x76543210);
      v16float out_flx_inter4 = fpselect16(
          fly_compare_ij, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      v8float flx_out4 =
          fpsub(flx_out3, out_flx_inter4, 0,
                0x76543210); // adds  fly_ijm -fly_ij + flx_imj -flx_ij

      // r1=R1, r2=R0

      v8float final_output =
          fpmul(concat(flx_out4, null_v8float()), 0, 0x76543210, flux_out_coeff,
                0, 0x00000000); // Multiply by +7s
      final_output =
          fpmac(final_output, data_buf2, 2, 0x76543210, coeffs1, 0, 0x76543210);
      // LOAD DATA FOR NEXT ITERATION
      row1_ptr = ((v8float *)(row1)) + i + 1;
      data_buf2 = upd_w(data_buf2, 0, *row1_ptr++);
      data_buf2 = upd_w(data_buf2, 1, *row1_ptr);
      *ptr_out++ = final_output;
    }
}

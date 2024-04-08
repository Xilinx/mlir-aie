//===- hdiff_lap_fp32.cc ----------------------------------------*- C++ -*-===//
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

void hdiff_lap_fp32(float *restrict row0, float *restrict row1,
                    float *restrict row2, float *restrict row3,
                    float *restrict row4, float *restrict out_flux1,
                    float *restrict out_flux2, float *restrict out_flux3,
                    float *restrict out_flux4) {

  // const float *restrict w = weights;
  alignas(32) float weights[8] = {-4, -4, -4, -4, -4, -4, -4, -4};
  alignas(32) float weights1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  alignas(32) float weights_rest[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  alignas(32) float flux_out[8] = {-7, -7, -7, -7, -7, -7, -7, -7};

  v8float coeffs = *(v8float *)weights;           //  8 x int32 = 256b W vector
  v8float coeffs1 = *(v8float *)weights1;         //  8 x int32 = 256b W vector
  v8float coeffs_rest = *(v8float *)weights_rest; //  8 x int32 = 256b W vector
  v8float flux_out_coeff = *(v8float *)flux_out;

  v8float *ptr_out = (v8float *)out_flux1;
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

  //  v8acc80 acc_1=null_v8acc80();
  v8float lap_ij = null_v8float(); //  8 x int32 = 256b W vector
  v8float lap_0 = null_v8float();  //  8 x int32 = 256b W vector

  data_buf1 = upd_w(data_buf1, 0, *row3_ptr++);
  data_buf1 = upd_w(data_buf1, 1, *row3_ptr);
  data_buf2 = upd_w(data_buf2, 0, *row1_ptr++);
  data_buf2 = upd_w(data_buf2, 1, *row1_ptr);

  for (unsigned i = 0; i < COL / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      v8float flux_sub;

      lap_ij =
          fpmul(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); ///  //c
      acc_1 = fpmul(data_buf2, 1, 0x76543210, coeffs_rest, 0, 0x00000000); // b

      lap_ij = fpmac(lap_ij, data_buf1, 2, 0x76543210, coeffs_rest, 0,
                     0x00000000); ///  //c,k
      acc_1 = fpmac(acc_1, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  //b,j

      row2_ptr = ((v8float *)(row2)) + i;
      data_buf2 = upd_w(data_buf2, 0, *(row2_ptr)++);
      data_buf2 = upd_w(data_buf2, 1, *(row2_ptr));

      /////////
      ///**************************LAP_ij**************************************
      lap_ij = fpmac(lap_ij, data_buf2, 1, 0x76543210, coeffs_rest, 0,
                     0x00000000); ///  //c,k,f
      lap_ij = fpmsc(lap_ij, data_buf2, 2, 0x76543210, coeffs, 0,
                     0x00000000); ///  //c,k,f,4*g
      lap_ij = fpmac(lap_ij, data_buf2, 3, 0x76543210, coeffs_rest, 0,
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
      ptr_out = (v8float *)out_flux1 + i;
      *ptr_out = flux_sub;
      //

      acc_0 = fpmul(data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); /// // l ; R1 is already loaded
      acc_0 = fpmsc(acc_0, data_buf2, 3, 0x76543210, coeffs, 0,
                    0x00000000); ///   // l, 4*h

      row1_ptr = ((v8float *)(row1)) + i;
      data_buf1 = upd_w(data_buf1, 0, *(row1_ptr)++);
      data_buf1 = upd_w(data_buf1, 1, *(row1_ptr));

      acc_0 = fpmac(acc_0, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  lap_ijp // l, 4*h, g
      acc_0 = fpmac(acc_0, data_buf2, 4, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  l, 4*h, g, i

      acc_0 = fpmac(acc_0, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  // l, 4*h, g, i, d

      flux_sub = fpsub(acc_0, concat(lap_ij, undef_v8float()), 0, 0x76543210);
      ptr_out = (v8float *)out_flux2 + i;
      *ptr_out = flux_sub;

      acc_1 = fpmul(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); // g
      acc_0 = fpmul(data_buf2, 2, 0x76543210, coeffs_rest, 0, 0x00000000); // g

      row0_ptr = ((v8float *)(row0)) + i;

      data_buf2 = upd_w(data_buf2, 0, *row0_ptr++);
      data_buf2 = upd_w(data_buf2, 1, *row0_ptr);

      acc_1 = fpmsc(acc_1, data_buf1, 2, 0x76543210, coeffs, 0,
                    0x00000000); // g, 4*c
      acc_1 = fpmac(acc_1, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, 4*c, b
      acc_1 = fpmac(acc_1, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); /// // g, 4*c, b, a

      row4_ptr = ((v8float *)(row4)) + i;

      data_buf2 = upd_w(data_buf2, 0, *row4_ptr++);
      data_buf2 = upd_w(data_buf2, 1, *row4_ptr);
      //////// **************************LAP_ipj for fly_ij since r2=R4********
      acc_1 = fpmac(acc_1, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); // g, 4*c, b, a, d
      acc_0 = fpmac(acc_0, data_buf2, 2, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///   // g, m

      // //Calculates lap_imj

      flux_sub = fpsub(lap_ij, concat(acc_1, undef_v8float()), 0, 0x76543210);
      ptr_out = (v8float *)out_flux3 + i;
      *ptr_out = flux_sub;

      row3_ptr = ((v8float *)(row3)) + i;

      data_buf1 = upd_w(data_buf1, 0, *row3_ptr++);
      data_buf1 = upd_w(data_buf1, 1, *row3_ptr);

      acc_0 = fpmsc(acc_0, data_buf1, 2, 0x76543210, coeffs, 0,
                    0x00000000); ///  //g, m , k * 4


      acc_0 = fpmac(acc_0, data_buf1, 1, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///  //g, m , k * 4, j

      // LOAD DATA FOR NEXT ITERATION

      row1_ptr = ((v8float *)(row1)) + i + 1;
      data_buf2 = upd_w(data_buf2, 0, *(row1_ptr)++);
      data_buf2 = upd_w(data_buf2, 1, *(row1_ptr));
      acc_0 = fpmac(acc_0, data_buf1, 3, 0x76543210, coeffs_rest, 0,
                    0x00000000); ///   //g, m , k * 4, j, l

      //  flx_ij = lap_ipj - lap_ij
      flux_sub = fpsub(acc_0, concat(lap_ij, undef_v8float()), 0, 0x76543210);
      ptr_out = (v8float *)out_flux4 + i;
      *ptr_out = flux_sub;

      // LOAD DATA FOR NEXT ITERATION
      row3_ptr = ((v8float *)(row3)) + i + 1;
      data_buf1 = upd_w(data_buf1, 0, *(row3_ptr)++);
      data_buf1 = upd_w(data_buf1, 1, *(row3_ptr));

    }
}

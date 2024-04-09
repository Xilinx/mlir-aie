//===- hdiff_flux2_fp32.cc --------------------------------------*- C++ -*-===//
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

void hdiff_flux2_fp32(float *restrict flux_inter1, float *restrict flux_inter2,
                      float *restrict flux_inter3, float *restrict flux_inter4,
                      float *restrict flux_inter5, float *restrict out) {

  alignas(32) float weights1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  alignas(32) float flux_out[8] = {-7, -7, -7, -7, -7, -7, -7, -7};

  v8float coeffs1 = *(v8float *)weights1; //  8 x int32 = 256b W vector
  v8float flux_out_coeff = *(v8float *)flux_out;

  v8float *restrict ptr_forward = (v8float *)flux_inter1;
  v8float *ptr_out = (v8float *)out;

  v16float data_buf2 = null_v16float();

  v8float acc_0 = null_v8float();
  v8float acc_1 = null_v8float();

  for (unsigned i = 0; i < COL / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      v8float flux_sub;
      v8float flux_interm_sub;

      ptr_forward = (v8float *)flux_inter1 + i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // compare > 0
      unsigned int flx_compare_imj =
          fpge(acc_1, null_v16float(), 0,
               0x76543210); /// flx_ijm * (test_in[d][c][r] -
                            /// test_in[d][c][r-1]) > 0 ? 0 :

      // //Calculate final fly_ijm
      v16float out_flx_inter1 = fpselect16(
          flx_compare_imj, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      v8float flx_out1 =
          fpadd(null_v8float(), out_flx_inter1, 0, 0x76543210); // still fly_ijm

      /////////////////////////////////////////////////////////////////////////////////////
      ptr_forward = (v8float *)flux_inter2 + i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // Calculates final fly_ij (comparison > 0)
      unsigned int flx_compare_ij = fpge(acc_1, null_v16float(), 0, 0x76543210);
      v16float out_flx_inter2 = fpselect16(
          flx_compare_ij, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      // fly_ijm -fly_ij
      v8float flx_out2 = fpsub(flx_out1, out_flx_inter2, 0, 0x76543210);
      /////////////////////////////////////////////////////////////////////////////////////
      ptr_forward = (v8float *)flux_inter3 + i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // Calculates final flx_imj (comparison > 0)
      unsigned int fly_compare_ijm =
          fpge(acc_1, null_v16float(), 0, 0x76543210);
      v16float out_flx_inter3 = fpselect16(
          fly_compare_ijm, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      v8float flx_out3 = fpadd(flx_out2, out_flx_inter3, 0,
                               0x76543210); // adds fly_ijm -fly_ij + flx_imj

      /////////////////////////////////////////////////////////////////////////////////////
      ptr_forward = (v8float *)flux_inter4 + i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // final flx_ij (comparison > 0 )
      unsigned int fly_compare_ij = fpge(acc_1, null_v16float(), 0, 0x76543210);
      v16float out_flx_inter4 = fpselect16(
          fly_compare_ij, concat(flux_sub, null_v8float()), 0, 0x76543210,
          0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);

      v8float flx_out4 =
          fpsub(flx_out3, out_flx_inter4, 0,
                0x76543210); // adds  fly_ijm -fly_ij + flx_imj -flx_ij

      ptr_forward = (v8float *)flux_inter5 + i;
      v8float tmp1 = *ptr_forward++;
      v8float tmp2 = *ptr_forward;
      data_buf2 = concat(tmp1, tmp2);

      v8float final_output =
          fpmul(concat(flx_out4, null_v8float()), 0, 0x76543210, flux_out_coeff,
                0, 0x00000000); // Multiply by +7s
      final_output =
          fpmac(final_output, data_buf2, 2, 0x76543210, coeffs1, 0, 0x76543210);

      *ptr_out++ = final_output;
    }
}

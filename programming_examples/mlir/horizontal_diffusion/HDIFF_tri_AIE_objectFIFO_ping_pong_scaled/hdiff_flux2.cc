//===- hdiff_flux2.cc -------------------------------------------*- C++ -*-===//
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

void hdiff_flux2(int32_t *restrict flux_inter1, int32_t *restrict flux_inter2,
                 int32_t *restrict flux_inter3, int32_t *restrict flux_inter4,
                 int32_t *restrict flux_inter5, int32_t *restrict out) {

  alignas(32) int32_t weights1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  alignas(32) int32_t flux_out[8] = {-7, -7, -7, -7, -7, -7, -7, -7};

  v8int32 coeffs1 = *(v8int32 *)weights1; //  8 x int32 = 256b W vector
  v8int32 flux_out_coeff = *(v8int32 *)flux_out;

  v8int32 *restrict ptr_forward = (v8int32 *)flux_inter1;
  v8int32 *ptr_out = (v8int32 *)out;

  v16int32 data_buf2 = null_v16int32();

  v8acc80 acc_0 = null_v8acc80();
  v8acc80 acc_1 = null_v8acc80();

  for (unsigned i = 0; i < COL / 8; i++)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      v8int32 flux_sub;
      v8int32 flux_interm_sub;

      ptr_forward = (v8int32 *)flux_inter1 + 2 * i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // compare > 0
      unsigned int flx_compare_imj =
          gt16(concat(flux_interm_sub, undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);

      // //Calculate final fly_ijm
      v16int32 out_flx_inter1 = select16(
          flx_compare_imj, concat(flux_sub, undef_v8int32()), null_v16int32());

      // still fly_ijm

      /////////////////////////////////////////////////////////////////////////////////////
      ptr_forward = (v8int32 *)flux_inter2 + 2 * i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // Calculates final fly_ij (comparison > 0)
      unsigned int flx_compare_ij =
          gt16(concat(flux_interm_sub, undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);
      v16int32 out_flx_inter2 = select16(
          flx_compare_ij, concat(flux_sub, undef_v8int32()), null_v16int32());

      // add fly_ij - fly_ijm
      v16int32 flx_out2 = sub16(out_flx_inter2, out_flx_inter1);
      /////////////////////////////////////////////////////////////////////////////////////
      ptr_forward = (v8int32 *)flux_inter3 + 2 * i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // Calculates final flx_imj (comparison > 0)
      unsigned int fly_compare_ijm =
          gt16(concat(flux_interm_sub, undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);
      v16int32 out_flx_inter3 = select16(
          fly_compare_ijm, concat(flux_sub, undef_v8int32()), null_v16int32());

      v16int32 flx_out3 =
          sub16(flx_out2, out_flx_inter3); // adds fly_ij - fly_ijm - flx_imj
      /////////////////////////////////////////////////////////////////////////////////////
      ptr_forward = (v8int32 *)flux_inter4 + 2 * i;
      flux_sub = *ptr_forward++;
      flux_interm_sub = *ptr_forward;

      // final flx_ij (comparison > 0 )
      unsigned int fly_compare_ij =
          gt16(concat(flux_interm_sub, undef_v8int32()), 0, 0x76543210,
               0xFEDCBA98, null_v16int32(), 0, 0x76543210, 0xFEDCBA98);
      v16int32 out_flx_inter4 = select16(
          fly_compare_ij, concat(flux_sub, undef_v8int32()), null_v16int32());

      v16int32 flx_out4 = add16(
          flx_out3, out_flx_inter4); // adds fly_ij - fly_ijm - flx_imj + flx_ij

      ptr_forward = (v8int32 *)flux_inter5 + 2 * i;
      v8int32 tmp1 = *ptr_forward++;
      v8int32 tmp2 = *ptr_forward;
      data_buf2 = concat(tmp2, tmp1);

      v8acc80 final_output = lmul8(flx_out4, 0, 0x76543210, flux_out_coeff, 0,
                                   0x00000000); // Multiply by -7s
      final_output = lmac8(final_output, data_buf2, 2, 0x76543210,
                           concat(coeffs1, undef_v8int32()), 0, 0x76543210);

      *ptr_out++ = srs(final_output, 0);
    }
}

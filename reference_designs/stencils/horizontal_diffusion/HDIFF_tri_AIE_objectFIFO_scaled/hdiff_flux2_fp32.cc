// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
  
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./include.h"
#include "hdiff.h"
#define kernel_load 14
// typedef int int32;


// void hdiff_flux(int32_t * restrict in, int32_t * restrict  flux_forward,  int32_t * restrict out)
// {
void hdiff_flux2_fp32(float* restrict flux_inter1,float* restrict flux_inter2, float* restrict flux_inter3,float* restrict flux_inter4,float* restrict flux_inter5, float * restrict out)
{
    
    alignas(32) float weights1[8] = {1,1,1,1,1,1,1,1};
    alignas(32) float flux_out[8]={-7,-7,-7,-7,-7,-7,-7,-7};

    v8float coeffs1        = *(v8float*) weights1;  //  8 x int32 = 256b W vector
    v8float flux_out_coeff = *(v8float*) flux_out;

    // v8int32 * restrict  ptr_in = (v8int32 *) in;
    v8float * restrict  ptr_forward = (v8float *) flux_inter1;
    v8float * ptr_out = (v8float *) out;
    


    // v8int32 * restrict row0_ptr=(v8int32 *)row0;
    // v8int32 * restrict row1_ptr=(v8int32 *)row1;
    // v8int32 * restrict row2_ptr=(v8int32 *)row2;

    // v16int32   data_buf1=null_v16int32();
    v16float  data_buf2=null_v16float();

    v8float acc_0 = null_v8float();
    v8float acc_1 = null_v8float();
          //  8 x int32 = 256b W vector

  
        // // r1 = ptr_in + 1*COL/8 + aor*COL/8;
        // // r2 = ptr_in + 2*COL/8 + aor*COL/8;
        // data_buf1 = upd_w(data_buf1, 0, *r1++);
        // data_buf1 = upd_w(data_buf1, 1, *r1);

        // data_buf2 = upd_w(data_buf2, 0, *r2++);
        // data_buf2 = upd_w(data_buf2, 1, *r2);

        
    // buf_2=R2 , and buf_1=R3
        
    for (unsigned i = 0; i < COL/8; i++)
        chess_prepare_for_pipelining
        chess_loop_range(1,)
        {
            v8float  flux_sub;
            v8float  flux_interm_sub;
            // printf("inside flux");
            // v8acc80 acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
            // flx_imj=window_readincr_v8(flux_cascade); //flx_imj
            ptr_forward = (v8float *) flux_inter1;
            flux_sub=*ptr_forward++;
            flux_interm_sub=*ptr_forward++;
       
            // compare > 0
            unsigned int flx_compare_imj=fpge (acc_1, null_v16float(), 0,0x76543210); /// flx_ijm * (test_in[d][c][r] - test_in[d][c][r-1]) > 0 ? 0 :
                         
            // //Calculate final fly_ijm
            v16float out_flx_inter1=fpselect16(flx_compare_imj,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
                  
            v8float flx_out1=fpadd(null_v8float(),out_flx_inter1,0,0x76543210);   //still fly_ijm

    /////////////////////////////////////////////////////////////////////////////////////
            // acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
            // flux_sub=window_readincr_v8(flux_cascade);
            ptr_forward = (v8float *) flux_inter2;
            flux_sub=*ptr_forward++;
            flux_interm_sub=*ptr_forward++;
    
            //Calculates final fly_ij (comparison > 0)
            unsigned int flx_compare_ij=fpge (acc_1, null_v16float(), 0,0x76543210); 
            v16float out_flx_inter2=fpselect16(flx_compare_ij,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
                        
                //fly_ijm -fly_ij
            v8float flx_out2=fpsub(flx_out1,out_flx_inter2,0,0x76543210);                          
/////////////////////////////////////////////////////////////////////////////////////

            /// retrieving flx_imj = lap_ij - lap_imj
            // acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
            // flux_sub=window_readincr_v8(flux_cascade);
            ptr_forward = (v8float *) flux_inter3;
            flux_sub=*ptr_forward++;
            flux_interm_sub=*ptr_forward++;

            //Calculates final flx_imj (comparison > 0)
            unsigned int fly_compare_ijm=fpge (acc_1, null_v16float(), 0,0x76543210);
            v16float out_flx_inter3=fpselect16(fly_compare_ijm,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
        
            v8float flx_out3=fpadd(flx_out2,out_flx_inter3,0,0x76543210);     //adds fly_ijm -fly_ij + flx_imj 

/////////////////////////////////////////////////////////////////////////////////////
             // acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
             //re†rieving flx_ij = lap_ipj - lap_ij
            // flux_sub=window_readincr_v8(flux_cascade);
            ptr_forward = (v8float *) flux_inter4;
            flux_sub=*ptr_forward++;
            flux_interm_sub=*ptr_forward++;

            // final flx_ij (comparison > 0 )
            unsigned int fly_compare_ij=fpge (acc_1, null_v16float(), 0,0x76543210);
            v16float out_flx_inter4=fpselect16(fly_compare_ij,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
            
            v8float flx_out4=fpsub(flx_out3,out_flx_inter4,0,0x76543210); //adds  fly_ijm -fly_ij + flx_imj -flx_ij
            
            ptr_forward = (v8float *) flux_inter5;
            v8float tmp1 =*ptr_forward++;
            v8float tmp2 = *ptr_forward++;
            data_buf2 = concat(tmp1,tmp2);

            v8float   final_output=fpmul  (concat(flx_out4,null_v8float()),0,0x76543210,flux_out_coeff,    0,0x00000000);  // Multiply by +7s
            final_output=fpmac(final_output,data_buf2,  2, 0x76543210,coeffs1, 0 , 0x76543210); 
            // window_writeincr(out, srs(final_output,0));
            *ptr_out++ =  final_output;  
                      
        }
    

}

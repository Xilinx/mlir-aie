/*  (c) Copyright 2014 - 2020 Xilinx, Inc. All rights reserved.

xchesscc -p me -P /proj/xbuilds/2021.1_released/installs/lin64/Vitis/2021.1/aietools/data/cervino/lib -L/proj/xbuilds/2021.1_released/installs/lin64/Vitis/2021.1/cardano/lib -c ./hdiff.cc
aiecc.py --sysroot=/group/xrlabs/platforms/pynq_on_versal_vck190_hacked/vck190-sysroot aie.mlir -v -I/scratch/gagandee/acdc-aie/runtime_lib/ /scratch/gagandee/acdc-aie/runtime_lib/test_library.cpp ./test.cpp -v -o test.elf
    
    
    
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


/* 
 * Kernel weighted_sum
 *
 * Compute the weighted sum of the last 8 values.
 */

// #include <adf.h>
#include "./include.h"
#include "hdiff.h"
#define kernel_load 14
// typedef int int32;


//align to 16 bytes boundary, equivalent to "alignas(v4int32)"
void vec_hdiff(float * restrict in, float * restrict out)
{

    alignas(32) float weights[8] = {-4,-4,-4,-4,-4,-4,-4,-4};
    alignas(32) float weights1[8] = {1,1,1,1,1,1,1,1};
    alignas(32) float weights_rest[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    alignas(32) float flux_out[8]={-7,-7,-7,-7,-7,-7,-7,-7};

    v8float coeffs     = *(v8float*) weights;  //  8 x int32 = 256b W vector
    v8float coeffs1     = *(v8float*) weights1;  //  8 x int32 = 256b W vector
    v8float coeffs_rest    = *(v8float*) weights_rest;  //  8 x int32 = 256b W vector
    v8float flux_out_coeff=*(v8float*)flux_out;
    
    v8float * restrict  ptr_in = (v8float *) in;
    v8float *   ptr_out = (v8float *) out;
    v8float * restrict r1=ptr_in+3*COL/8;
    v8float * restrict r2=ptr_in+1*COL/8;
    // v16int32 * restrict r3=ptr_in+3*COL/4;
    // v16int32 * restrict r4=ptr_in+4*COL/4;
    // v16int32 * restrict r5=ptr_in+5*COL/4;
    // v32int32 data_buf=undef_v32int32();
    v16float   data_buf1=null_v16float();
    v16float  data_buf2=null_v16float();
    
    v8float acc_0=null_v8float();
    v8float acc_1=null_v8float();
                //  v8acc80 acc_1=null_v8acc80();        
    v8float  lap_ij  = null_v8float();      //  8 x int32 = 256b W vector
    v8float lap_0    = null_v8float();      //  8 x int32 = 256b W vector

    for(unsigned aor=0; aor < kernel_load; aor++)
    {
  
        r1 = ptr_in + 3*COL/8 + aor*COL/8;
        r2 = ptr_in + 1*COL/8 + aor*COL/8;

        data_buf1 = upd_w(data_buf1, 0, *r1++);
        data_buf1 = upd_w(data_buf1, 1, *r1);
        data_buf2 = upd_w(data_buf2, 0, *r2++);
        data_buf2 = upd_w(data_buf2, 1, *r2);

    

        for (unsigned i = 0; i < COL/8; i++)
            chess_prepare_for_pipelining
                    chess_loop_range(1,)
            {   v8float flux_sub;
                               
                // buf_2=R1, and buf_1=R3
                ///////// **************************LAP_ij**************************************
                lap_ij=fpmul (data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);///  //c      
                ///////// **************************LAP_ijm**************************************
                acc_1=fpmul(data_buf2,1,0x76543210,coeffs_rest,    0,0x00000000);        //b
            
                    ///////// **************************LAP_ij**************************************
                lap_ij=fpmac   (lap_ij,data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000);///  //c,k 
                ///////// **************************LAP_imj**************************************
                acc_1=fpmac   (acc_1,data_buf1,1,0x76543210,coeffs_rest,    0,0x00000000);///  //b,j
                

                r2 = ptr_in+2 * COL/8+i + aor*COL/8;
   
                data_buf2 = upd_w(data_buf2, 0, *r2++);
                data_buf2 = upd_w(data_buf2, 1, *r2);
      
                ///////// **************************LAP_ij**************************************
                lap_ij=fpmac   (lap_ij,data_buf2,1,0x76543210,coeffs_rest,    0,0x00000000);///  //c,k,f
                lap_ij=fpmsc   (lap_ij,data_buf2,2,0x76543210,coeffs,    0,0x00000000);///  //c,k,f,4*g
                lap_ij=fpmac   (lap_ij,data_buf2,3,0x76543210,coeffs_rest,    0,0x00000000);///  c,k,f,4*g,h
                    ///////// **************************LAP_ijm**************************************               
                acc_1=fpmac   (acc_1,data_buf2,0,0x76543210,coeffs_rest,    0,0x00000000);///  //b,j,e         
                acc_1=fpmsc   (acc_1,data_buf2,1,0x76543210,coeffs,    0,0x00000000);///         //b,j,e,4*f   
                acc_1=fpmac   (acc_1,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);///     //b,j,e,4*f,g  
               
                //Calculate  lap_ij - lap_ijm
                flux_sub = fpsub(lap_ij, concat(acc_1,undef_v8float()), 0, 0x76543210); // flx_imj = lap_ij - lap_ijm;
                

                 //*****************FLUX //below reuisng acc_1 for flux calculation
                acc_1=fpmul   (data_buf2,2,0x76543210,flux_sub,    0,0x00000000);///  (lap_ij - lap_ijm)*g
                acc_1=fpmsc   (acc_1,data_buf2,1,0x76543210,flux_sub,    0,0x00000000);// (lap_ij - lap_ijm)*g - (lap_ij - lap_ijm)*f

                // compare > 0
                unsigned int flx_compare_imj=fpge (acc_1, null_v16float(), 0,0x76543210); /// flx_ijm * (test_in[d][c][r] - test_in[d][c][r-1]) > 0 ? 0 :
                
                    // ///////// **************************lap_ipj flx_ij**************************************
                acc_0=fpmul   (data_buf1,3,0x76543210,coeffs_rest,    0,0x00000000);/// // l ; R1 is already loaded
                // ///////// **************************lap_ipj flx_ij**************************************
                acc_0=fpmsc   (acc_0,data_buf2,3,0x76543210,coeffs,    0,0x00000000);///   // l, 4*h

               //Calculate final fly_ijm
                v16float out_flx_inter1=fpselect16(flx_compare_imj,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
                  
                r1 = ptr_in+1 * COL/8+i + aor*COL/8;
                // data_buf1=*r1++;     
                data_buf1 = upd_w(data_buf1, 0, *r1++);
                data_buf1 = upd_w(data_buf1, 1, *r1); 

                acc_0=fpmac  (acc_0,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);///  lap_ijp // l, 4*h, g
                acc_0=fpmac   (acc_0,data_buf2,4,0x76543210,coeffs_rest,    0,0x00000000);///  l, 4*h, g, i   
                v8float flx_out1=fpadd(null_v8float(),out_flx_inter1,0,0x76543210);   //still fly_ijm
                
               
                acc_0=fpmac   (acc_0,data_buf1,3,0x76543210,coeffs_rest,    0,0x00000000);///  // l, 4*h, g, i, d   
                
                //Calculates lap_ijp - lap_ij
                flux_sub = fpsub(acc_0, concat(lap_ij,undef_v8float()), 0, 0x76543210); 
               
                acc_1=fpmul   (data_buf2,3,0x76543210,flux_sub,    0,0x00000000); // (lap_ijp - lap_ij) * h  
                acc_1=fpmsc   (acc_1,data_buf2,2,0x76543210,flux_sub,    0,0x00000000); //  (lap_ijp - lap_ij) * h  - (lap_ijp - lap_ij) * g                     

                //Calculates final fly_ij (comparison > 0)
                unsigned int flx_compare_ij=fpge (acc_1, null_v16float(), 0,0x76543210); 
                v16float out_flx_inter2=fpselect16(flx_compare_ij,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
                        
                //fly_ijm -fly_ij
                v8float flx_out2=fpsub(flx_out1,out_flx_inter2,0,0x76543210);
                //NOTE: fbsub does not support v16-v8 so instead of calculating fly_ij - fly_ijm - flx_imj + flx_ij
                //              we calculate -(-fly_ij + fly_ijm + flx_imj - flx_ij)
                // Therefore, flux coeff is positive as well.
 
    // //***********************************************************************STARTING X FLUX*****************************************************************************************************************************************************
            //            //////// **************************LAP_imj sincer2=R2 are already loaded********
                acc_1=fpmul    ( data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);        // g  
                        
                //////// **************************LAP_ipj for fly_ij sincer2=R2 are already loaded********
                acc_0=fpmul   (data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);// g   


                r2 = ptr_in + 0*COL/8 + i + aor*COL/8;  // load for LAP_ijm for fly_ijm 
                data_buf2 = upd_w(data_buf2, 0, *r2++);
                data_buf2 = upd_w(data_buf2, 1, *r2);

                    // ///////// **************************LAP_imj since R1=buff 1 ************************************
        
                acc_1=fpmsc    (acc_1, data_buf1,2,0x76543210,coeffs,    0,0x00000000);      // g, 4*c
                acc_1=fpmac    (acc_1, data_buf1,1,0x76543210,coeffs_rest,    0,0x00000000);       // g, 4*c, b
                acc_1=fpmac   (acc_1,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);/// // g, 4*c, b, a
            
                r2 = ptr_in + 4*COL/8 + i + aor*COL/8;  // load for LAP_ijm for fly_ijm 
                data_buf2 = upd_w(data_buf2, 0, *r2++);
                data_buf2 = upd_w(data_buf2, 1, *r2);
                    //////// **************************LAP_ipj for fly_ij since r2=R4********
                acc_1=fpmac    ( acc_1,data_buf1,3,0x76543210,coeffs_rest,    0,0x00000000);       // g, 4*c, b, a, d  
                acc_0=fpmac   (acc_0,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);///   // g, m
                
                r2 = ptr_in + 2*COL/8 + i + aor*COL/8;  // load for LAP_ijm for fly_ijm 
                data_buf2 = upd_w(data_buf2, 0, *r2++);
                data_buf2 = upd_w(data_buf2, 1, *r2);
                
                
                //flx_imj = lap_ij - lap_imj
                flux_sub = fpsub(lap_ij, concat(acc_1,undef_v8float()), 0, 0x76543210); 
               
                acc_1=fpmul   (data_buf2,2,0x76543210,flux_sub,    0,0x00000000);///   (lap_ij - lap_imj) * g
                acc_1=fpmsc   (acc_1,data_buf1,2,0x76543210,flux_sub,    0,0x00000000);///    (lap_ij - lap_imj) * g  *  (lap_ij - lap_imj) * c          
                
                //Calculates final flx_imj (comparison > 0)
                unsigned int fly_compare_ijm=fpge (acc_1, null_v16float(), 0,0x76543210);
                v16float out_flx_inter3=fpselect16(fly_compare_ijm,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
        

                r1 = ptr_in + 3*COL/8 + i + aor*COL/8; // load for LAP_ijp for fly_ij
                // data_buf1=*r1++;
                data_buf1 = upd_w(data_buf1, 0, *r1++);
                data_buf1 = upd_w(data_buf1, 1, *r1);
                ////// **************************since r1=R1********
                acc_0=fpmsc   (acc_0,data_buf1,2,0x76543210,coeffs,    0,0x00000000);///  //g, m , k * 4

                v8float flx_out3=fpadd(flx_out2,out_flx_inter3,0,0x76543210);     //adds fly_ijm -fly_ij + flx_imj 

                acc_0=fpmac   (acc_0,data_buf1, 1,0x76543210,coeffs_rest,    0,0x00000000);///  //g, m , k * 4, j   
                acc_0=fpmac   (acc_0,data_buf1, 3,0x76543210,coeffs_rest,    0,0x00000000);///   //g, m , k * 4, j, l   
                // lap_0=srs(acc_0,0); //LAP_ijp
                ///////// **************************fly_ij *************************************
                //  flx_ij = lap_ipj - lap_ij
                flux_sub = fpsub(acc_0, concat(lap_ij,undef_v8float()), 0, 0x76543210);
           
                //below reuisng acc_1 for flux calculation //CHANGED FROM 3 TO 2
                acc_1=fpmul   (data_buf1,2,0x76543210,flux_sub,    0,0x00000000);//  (lap_ipj - lap_ij) * k       
                
                //LOAD DATA FOR NEXT ITERATION
                r1 = ptr_in + 3*COL/8 + i + 1 + aor*COL/8;
                // data_buf1=*r1++;
                data_buf1 = upd_w(data_buf1, 0, *r1++);
                data_buf1 = upd_w(data_buf1, 1, *r1);
                
                acc_1=fpmsc   (acc_1,data_buf2,2,0x76543210,flux_sub,    0,0x00000000); //  (lap_ipj - lap_ij) * k - (lap_ipj - lap_ij) * g   

                 // final flx_ij (comparison > 0 )
                unsigned int fly_compare_ij=fpge (acc_1, null_v16float(), 0,0x76543210);
                v16float out_flx_inter4=fpselect16(fly_compare_ij,concat(flux_sub,null_v8float()),0, 0x76543210, 0xFEDCBA98, null_v16float(), 0, 0x76543210, 0xFEDCBA98);
                
                v8float flx_out4=fpsub(flx_out3,out_flx_inter4,0,0x76543210); //adds  fly_ijm -fly_ij + flx_imj -flx_ij
               
                // r1=R1, r2=R0
           
                v8float   final_output=fpmul  (concat(flx_out4,null_v8float()),0,0x76543210,flux_out_coeff,    0,0x00000000);  // Multiply by +7s
                final_output=fpmac(final_output,data_buf2,  2, 0x76543210,coeffs1, 0 , 0x76543210); 
                //LOAD DATA FOR NEXT ITERATION
                r2 = ptr_in + 1*COL/8 + i + 1 + aor*COL/8;
                data_buf2 = upd_w(data_buf2, 0, *r2++);
                data_buf2 = upd_w(data_buf2, 1, *r2);

                *ptr_out++ = final_output;       
                //  window_writeincr(out, final_output);        

            }
    }   
}

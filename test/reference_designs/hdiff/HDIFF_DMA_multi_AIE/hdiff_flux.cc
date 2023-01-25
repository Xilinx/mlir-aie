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


void hdiff_flux(int32_t * restrict in, int32_t * restrict  flux_forward,  int32_t * restrict out)
{
   
    
    alignas(32) int32_t weights1[8] = {1,1,1,1,1,1,1,1};
    alignas(32) int32_t flux_out[8]={-7,-7,-7,-7,-7,-7,-7,-7};

    v8int32 coeffs1         = *(v8int32*) weights1;  //  8 x int32 = 256b W vector
    v8int32 flux_out_coeff=*(v8int32*)flux_out;

    v8int32 * restrict  ptr_in = (v8int32 *) in;
    v8int32 * restrict  ptr_forward = (v8int32 *) flux_forward;
    v8int32 * ptr_out = (v8int32 *) out;
    
    v8int32 * restrict r1=ptr_in+1*COL/8;
    v8int32 * restrict r2=ptr_in+2*COL/8;

    v16int32   data_buf1=null_v16int32();
    v16int32  data_buf2=null_v16int32();

    v8acc80 acc_0=null_v8acc80();
    v8acc80 acc_1=null_v8acc80();
          //  8 x int32 = 256b W vector

    for(unsigned aor=0; aor < kernel_load; aor++)
    {
        r1 = ptr_in + 1*COL/8 + aor*COL/8;
        r2 = ptr_in + 2*COL/8 + aor*COL/8;
        data_buf1 = upd_w(data_buf1, 0, *r1++);
        data_buf1 = upd_w(data_buf1, 1, *r1);

        data_buf2 = upd_w(data_buf2, 0, *r2++);
        data_buf2 = upd_w(data_buf2, 1, *r2);

        
    // buf_2=R2 , and buf_1=R3
        
    for (unsigned i = 0; i < COL/8; i++)
        chess_prepare_for_pipelining
        chess_loop_range(1,)
        {
            v8int32  flux_sub;
            // printf("inside flux");
            // v8acc80 acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
            // flx_imj=window_readincr_v8(flux_cascade); //flx_imj
            flux_sub=*ptr_forward++;
            // flux_sub=window_readincr_v8(flux_cascade);
            acc_1=lmul8   (data_buf2,2,0x76543210,flux_sub,    0,0x00000000);           // (lap_ij - lap_ijm)*g
            acc_1=lmsc8   (acc_1,data_buf2,1,0x76543210,flux_sub,    0,0x00000000);     // (lap_ij - lap_ijm)*g - (lap_ij - lap_ijm)*f

            // compare > 0
            unsigned int flx_compare_imj=gt16(concat(srs(acc_1,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
                        
            // //Calculate final fly_ijm
            v16int32 out_flx_inter1=select16(flx_compare_imj,concat(flux_sub,undef_v8int32()),null_v16int32()); 

            v16int32 flx_out1=add16(null_v16int32(),out_flx_inter1);     //still fly_ijm
                
    /////////////////////////////////////////////////////////////////////////////////////
            // acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
            // flux_sub=window_readincr_v8(flux_cascade);
            flux_sub=*ptr_forward++;
            
            acc_0=lmul8   (data_buf2,3,0x76543210,flux_sub,    0,0x00000000);            // (lap_ijp - lap_ij) * h       
            acc_0=lmsc8   (acc_0,data_buf2,2,0x76543210,flux_sub,    0,0x00000000);      //  (lap_ijp - lap_ij) * h  - (lap_ijp - lap_ij) * g           

            //Calculates final fly_ij (comparison > 0)
            unsigned int flx_compare_ij=gt16(concat(srs(acc_0,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
            v16int32 out_flx_inter2=select16(flx_compare_ij,concat(flux_sub,undef_v8int32()),null_v16int32());    

            //add fly_ij - fly_ijm
            v16int32 flx_out2=sub16(out_flx_inter2,flx_out1);                             
/////////////////////////////////////////////////////////////////////////////////////

            /// retrieving flx_imj = lap_ij - lap_imj
            // acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
            // flux_sub=window_readincr_v8(flux_cascade);
            flux_sub=*ptr_forward++;
            acc_1=lmul8   (data_buf2,2,0x76543210,flux_sub,    0,0x00000000);       //   (lap_ij - lap_imj) * g
            acc_1=lmsc8   (acc_1,data_buf1,2,0x76543210,flux_sub,    0,0x00000000); //    (lap_ij - lap_imj) * g  *  (lap_ij - lap_imj) * c  
            
            r1 = ptr_in + 3*COL/8 + i + aor*COL/8;
            data_buf1 = upd_w(data_buf1, 0, *r1++);
            data_buf1 = upd_w(data_buf1, 1, *r1);
            //Calculates final flx_imj (comparison > 0)
            unsigned int fly_compare_ijm=gt16(concat(srs(acc_1,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
            v16int32 out_flx_inter3=select16(fly_compare_ijm,concat(flux_sub,undef_v8int32()),null_v16int32());

            v16int32 flx_out3=sub16(flx_out2,out_flx_inter3);                     //adds fly_ij - fly_ijm - flx_imj
/////////////////////////////////////////////////////////////////////////////////////
             // acc_flux = concat(get_scd_v4acc80(),get_scd_v4acc80());
             //re†rieving flx_ij = lap_ipj - lap_ij
            // flux_sub=window_readincr_v8(flux_cascade);
            flux_sub=*ptr_forward++;

            //below reuisng acc_1 for flux calculation //CHANGED FROM 3 TO 2
            acc_1=lmul8   (data_buf1,2,0x76543210,flux_sub,    0,0x00000000);      //  (lap_ipj - lap_ij) * k       

            acc_1=lmsc8   (acc_1,data_buf2,2,0x76543210,flux_sub,    0,0x00000000);    //  (lap_ipj - lap_ij) * k - (lap_ipj - lap_ij) * g   

            
            r1 = ptr_in + 1*COL/8 + i+1+ aor*COL/8;
            data_buf1 = upd_w(data_buf1, 0, *r1++);
            data_buf1 = upd_w(data_buf1, 1, *r1);
     

            // final flx_ij (comparison > 0 )
            unsigned int fly_compare_ij=gt16(concat(srs(acc_1,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
            v16int32 out_flx_inter4=select16(fly_compare_ij,concat(flux_sub,undef_v8int32()),null_v16int32()); 

            v16int32 flx_out4=add16(flx_out3,out_flx_inter4); //adds fly_ij - fly_ijm - flx_imj + flx_ij

            v8acc80 final_output = lmul8  (flx_out4, 0, 0x76543210, flux_out_coeff,    0,0x00000000);  // Multiply by -7s
            final_output=lmac8(final_output, data_buf2,  2, 0x76543210,concat(coeffs1, undef_v8int32()), 0 , 0x76543210); 


          //LOAD DATA FOR NEXT ITERATION
            
            r2 = ptr_in + 2*COL/8 + i + 1 + aor*COL/8;
            // data_buf1=*r1++;
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);

            // window_writeincr(out, srs(final_output,0));
            *ptr_out++ =  srs(final_output,0);  
                      
        }
    }

}

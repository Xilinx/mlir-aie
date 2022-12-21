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
void vec_hdiff(int32_t* restrict in, int32_t* restrict out)
{

 // const int32_t *restrict w = weights;
    alignas(32) int32_t weights[8] = {-4,-4,-4,-4,-4,-4,-4,-4};
    alignas(32) int32_t weights1[8] = {1,1,1,1,1,1,1,1};
    alignas(32) int32_t weights_rest[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    alignas(32) int32_t flux_out[8]={-7,-7,-7,-7,-7,-7,-7,-7};

    v8int32 coeffs         = *(v8int32*) weights;  //  8 x int32 = 256b W vector
    v8int32 coeffs1        = *(v8int32*) weights1;  //  8 x int32 = 256b W vector
    v8int32 coeffs_rest    = *(v8int32*) weights_rest;  //  8 x int32 = 256b W vector
    v8int32 flux_out_coeff = *(v8int32*) flux_out;

    v8int32 * restrict ptr_in = (v8int32 *) in;
    v8int32 * ptr_out = (v8int32 *) out;
    
    v8int32 * restrict r1=ptr_in+3*COL/8;
    v8int32 * restrict r2=ptr_in+1*COL/8;

    v16int32 data_buf1 = null_v16int32();
    v16int32 data_buf2 = null_v16int32();;
    
    v8acc80 acc_0 = null_v8acc80();
    v8acc80 acc_1 = null_v8acc80();

    //  v8acc80 acc_1=null_v8acc80();        
    v8int32 lap_ij = null_v8int32();      //  8 x int32 = 256b W vector
    v8int32 lap_0  = null_v8int32();      //  8 x int32 = 256b W vector

    // The following loop is our sliding window. By Advancing One Row (aor) each time, we can process a new row:

    // This is how it works:
    // Input is 265*265*64. For each output row, we need five input rows. By adding two zero-rows, we pad the input to be 260*256*64
    // Then, we can split the work between multiple cores. For example, dividing the work between 16 cores would look like this:
    // 260*256 as input will produce 256*256 output. In a 16-cores setup, each core will produce 16 (kernel_load) output rows:
    //          input rows       output rows
    // core #1      0~19              0~15
    // core #2     16~35             16~31
    // core #3     32~51             32~47
    //  ...         ...               ...
    // core #16   240~259           240~255
    //
    // This means that cores #n and #n+1 have four common rows in their inputs. For example for cores #1 and #2, input rows 16~19 are common, and are copied twice.
    // In total, we have to copy (#cores-1)x4 rows twice. On the other hand, we can only handle 28,672 (0x7000) bytes in each tile, or 7168 int32. 
    // This is equal to a total of 28 rows including ALL the buffers: input's ping and pong and output's ping and pong. If each input buffer include X rows, each output buffer will
    // include X-4 rows. Hence, each tile can have maximum 2X + 2(X-4) = 28 rows, or X = 9. 
    // Which means that between input's ping and pong buffers, we always have to copy 4/9 rows twice (44%).
    // The other approach is to replace the ping pong buffers with single buffers: X + (X-4) = 28, or X = 16. In this case, we copy 4/16 rows twice (25%).
    // If we can utulize neighboring tiles' memory, we can do X + (X-4) = 4*28, or X = 58 and 4/58 will be 6% redundancy.
     
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
        {   
            v16int32 flux_sub;

            acc_0=lmul8   (data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);          //c           
            acc_1=lmul8   (data_buf2,1,0x76543210,coeffs_rest,    0,0x00000000);          //b
          
            acc_0=lmac8   (acc_0,data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000);    //c,k
            acc_1=lmac8   (acc_1,data_buf1,1,0x76543210,coeffs_rest,    0,0x00000000);    //b,j
          
            r2 = ptr_in+2 * COL/8+i + aor*COL/8;
            
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);

            acc_0=lmac8   (acc_0,data_buf2,1,0x76543210,coeffs_rest,    0,0x00000000); //c,k,f
            acc_0=lmsc8   (acc_0,data_buf2,2,0x76543210,coeffs,    0,0x00000000);      //c,k,f,4*g
            acc_0=lmac8   (acc_0,data_buf2,3,0x76543210,coeffs_rest,    0,0x00000000);  //c,k,f,4*g,h
            
            lap_ij=srs(acc_0,0); //store lap_ij

            acc_1=lmac8   (acc_1,data_buf2,0,0x76543210,coeffs_rest,    0,0x00000000); //b,j,e
            acc_1=lmsc8   (acc_1,data_buf2,1,0x76543210,coeffs,    0,0x00000000);      //b,j,e,4*f
            acc_1=lmac8   (acc_1,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);  //b,j,e,4*f,g  
           
            //lap_ijm
            lap_0=srs(acc_1,0);
    
            //Calculate  lap_ij - lap_ijm
            flux_sub = sub16(concat(lap_ij,undef_v8int32()), 0, 0x76543210, 0xFEDCBA98,  concat(lap_0,undef_v8int32()), 0,  0x76543210, 0xFEDCBA98 ); 
            
            //
            acc_1=lmul8   (data_buf2,2,0x76543210,ext_w(flux_sub,0),    0,0x00000000);           // (lap_ij - lap_ijm)*g
            acc_1=lmsc8   (acc_1,data_buf2,1,0x76543210,ext_w(flux_sub,0),    0,0x00000000);     // (lap_ij - lap_ijm)*g - (lap_ij - lap_ijm)*f

            // compare > 0
            unsigned int flx_compare_imj=gt16(concat(srs(acc_1,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
            
            acc_0=lmul8   (data_buf1,3,0x76543210,coeffs_rest,    0,0x00000000);   // l
            acc_0=lmsc8   (acc_0,data_buf2,3,0x76543210,coeffs,    0,0x00000000);  // l, 4*h
            
            //Calculate final fly_ijm
            v16int32 out_flx_inter1=select16(flx_compare_imj,flux_sub,null_v16int32()); 

            r1 = ptr_in+1 * COL/8+i + aor*COL/8;
            data_buf1 = upd_w(data_buf1, 0, *r1++);
            data_buf1 = upd_w(data_buf1, 1, *r1);    

            acc_0=lmac8  (acc_0,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);  // l, 4*h, g
            acc_0=lmac8   (acc_0,data_buf2,4,0x76543210,coeffs_rest,    0,0x00000000); // l, 4*h, g, i 
            v16int32 flx_out1=add16(null_v16int32(),out_flx_inter1);     //still fly_ijm
                      
            acc_0=lmac8   (acc_0,data_buf1,3,0x76543210,coeffs_rest,    0,0x00000000);// l, 4*h, g, i, d 
            
            //Calculates lap_ijp
            lap_0=srs(acc_0,0); 

            //Calculates lap_ijp - lap_ij
            flux_sub = sub16(concat(lap_0,undef_v8int32()), 0, 0x76543210, 0xFEDCBA98,  concat(lap_ij,undef_v8int32()), 0,  0x76543210, 0xFEDCBA98 );
            
            acc_0=lmul8   (data_buf2,3,0x76543210,ext_w(flux_sub,0),    0,0x00000000);            // (lap_ijp - lap_ij) * h       
            acc_0=lmsc8   (acc_0,data_buf2,2,0x76543210,ext_w(flux_sub,0),    0,0x00000000);      //  (lap_ijp - lap_ij) * h  - (lap_ijp - lap_ij) * g           

            //Calculates final fly_ij (comparison > 0)
            unsigned int flx_compare_ij=gt16(concat(srs(acc_0,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
            v16int32 out_flx_inter2=select16(flx_compare_ij,flux_sub,null_v16int32());    

            //add fly_ij - fly_ijm
            v16int32 flx_out2=sub16(out_flx_inter2,flx_out1);                             

            //***********************************************************************STARTING X FLUX*****************************************************************************************************************************************************
            
            acc_1=lmul8    ( data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000); // g                     
            acc_0=lmul8   (data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);   // g

            r2 = ptr_in + 0*COL/8 + i + aor*COL/8; 
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);
                  
            acc_1=lmsc8    (acc_1, data_buf1,2,0x76543210,coeffs,    0,0x00000000);             // g, 4*c
            acc_1=lmac8    (acc_1, data_buf1,1,0x76543210,coeffs_rest,    0,0x00000000);        // g, 4*c, b
            acc_1=lmac8   (acc_1,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);          // g, 4*c, b, a
          
            r2 = ptr_in + 4*COL/8 + i + aor*COL/8; 
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);

            acc_1=lmac8    ( acc_1,data_buf1,3,0x76543210,coeffs_rest,    0,0x00000000);       // g, 4*c, b, a, d               
            acc_0=lmac8   (acc_0,data_buf2,2,0x76543210,coeffs_rest,    0,0x00000000);         // g, m
            
            r2 = ptr_in + 2*COL/8 + i + aor*COL/8; 
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);
            
            //Calculates lap_imj
            lap_0=srs(acc_1,0); 

            //flx_imj = lap_ij - lap_imj
            flux_sub = sub16(concat(lap_ij,undef_v8int32()), 0, 0x76543210, 0xFEDCBA98,  concat(lap_0,undef_v8int32()), 0,  0x76543210, 0xFEDCBA98 ); 
            
            acc_1=lmul8   (data_buf2,2,0x76543210,ext_w(flux_sub,0),    0,0x00000000);       //   (lap_ij - lap_imj) * g
            acc_1=lmsc8   (acc_1,data_buf1,2,0x76543210,ext_w(flux_sub,0),    0,0x00000000); //    (lap_ij - lap_imj) * g  *  (lap_ij - lap_imj) * c  
     
            //Calculates final flx_imj (comparison > 0)
            unsigned int fly_compare_ijm=gt16(concat(srs(acc_1,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
            v16int32 out_flx_inter3=select16(fly_compare_ijm,flux_sub,null_v16int32());

            r1 = ptr_in + 3*COL/8 + i + aor*COL/8;
            data_buf1 = upd_w(data_buf1, 0, *r1++);
            data_buf1 = upd_w(data_buf1, 1, *r1);

            acc_0=lmsc8   (acc_0,data_buf1,2,0x76543210,coeffs,    0,0x00000000); //g, m , k * 4

            v16int32 flx_out3=sub16(flx_out2,out_flx_inter3);                     //adds fly_ij - fly_ijm - flx_imj

            acc_0=lmac8   (acc_0,data_buf1, 1,0x76543210,coeffs_rest,    0,0x00000000);     //g, m , k * 4, j 
            acc_0=lmac8   (acc_0,data_buf1, 3,0x76543210,coeffs_rest,    0,0x00000000);     //g, m , k * 4, j, l  
     
            //  flx_ij = lap_ipj - lap_ij
            flux_sub = sub16(concat(srs(acc_0,0),undef_v8int32()), 0, 0x76543210, 0xFEDCBA98, concat(lap_ij,undef_v8int32()),   0,  0x76543210, 0xFEDCBA98 ); 

            //below reuisng acc_1 for flux calculation //CHANGED FROM 3 TO 2
            acc_1=lmul8   (data_buf1,2,0x76543210,ext_w(flux_sub,0),    0,0x00000000);      //  (lap_ipj - lap_ij) * k       
            
            //LOAD DATA FOR NEXT ITERATION
            r1 = ptr_in + 3*COL/8 + i + 1 + aor*COL/8;
            data_buf1 = upd_w(data_buf1, 0, *r1++);
            data_buf1 = upd_w(data_buf1, 1, *r1);
            
            acc_1=lmsc8   (acc_1,data_buf2,2,0x76543210,ext_w(flux_sub,0),    0,0x00000000);    //  (lap_ipj - lap_ij) * k - (lap_ipj - lap_ij) * g   

            // final flx_ij (comparison > 0 )
            unsigned int fly_compare_ij=gt16(concat(srs(acc_1,0),undef_v8int32()),0,0x76543210,0xFEDCBA98, null_v16int32(),0,0x76543210,0xFEDCBA98); 
            v16int32 out_flx_inter4=select16(fly_compare_ij,flux_sub,null_v16int32()); 

            v16int32 flx_out4=add16(flx_out3,out_flx_inter4); //adds fly_ij - fly_ijm - flx_imj + flx_ij

            v8acc80 final_output = lmul8  (flx_out4, 0, 0x76543210, flux_out_coeff,    0,0x00000000);  // Multiply by -7s
            final_output=lmac8(final_output, data_buf2,  2, 0x76543210,concat(coeffs1, undef_v8int32()), 0 , 0x76543210); 

            //LOAD DATA FOR NEXT ITERATION
            r2 = ptr_in + 1*COL/8 + i + 1 + aor*COL/8;
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);

            // window_writeincr(out, srs(final_output,0));
            *ptr_out++ =  srs(final_output,0);       
        }

    }
}

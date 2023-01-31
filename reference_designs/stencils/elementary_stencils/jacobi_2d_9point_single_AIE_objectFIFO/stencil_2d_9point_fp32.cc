// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
  
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./include.h"
#include "stencil.h"
#define kernel_load 14
// typedef int int32;


//align to 16 bytes boundary, equivalent to "alignas(v4int32)"
void stencil_2d_9point_fp32(float* restrict in0, float* restrict in1,float* restrict in2,float* restrict in3, float* restrict in4,float* restrict out)
{
 // const float *restrict w = weights;
    alignas(32) float weights[8] = {-12,-12,-12,-12,-12,-12,-12,-12};
    alignas(32) float weights_rest[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
   
    v8float coeffs         = *(v8float*) weights;  //  8 x int32 = 256b W vector
    v8float coeffs_rest    = *(v8float*) weights_rest;  //  8 x int32 = 256b W vector


    // v8float * restrict ptr_in = (v8float *) in->ptr;
    v8float * ptr_out = (v8float *) out;
    v8float * restrict r0=(v8float *) in0;
    v8float * restrict r1=(v8float *) in1;
    v8float * restrict r2=(v8float *) in2;
    v8float * restrict r3=(v8float *) in3;
    v8float * restrict r4=(v8float *) in4;

    v16float data_buf1 = null_v16float();
    v16float data_buf2 = null_v16float();

    v8float acc_0 = null_v8float();

    data_buf1 = upd_w(data_buf1, 0, *r0++);
    data_buf1 = upd_w(data_buf1, 1, *r0);

    data_buf2 = upd_w(data_buf2, 0, *r2++);
    data_buf2 = upd_w(data_buf2, 1, *r2);
    
        for (unsigned i = 0; i < COL/8; i++)
            chess_prepare_for_pipelining
                    chess_loop_range(1,)
            {
            acc_0=fpmul   (data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000);      
            acc_0=fpmac   (acc_0,data_buf2,0,0x76543210,coeffs_rest,    0,0x00000000); 

            r1=((v8float *) in1)+i;
            data_buf1 = upd_w(data_buf1, 0, *r1++);
            data_buf1 = upd_w(data_buf1, 1, *r1);
            
            acc_0=fpmac   (acc_0,data_buf2,1,0x76543210,coeffs_rest,    0,0x00000000);  
            acc_0=fpmac   (acc_0,data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000);

            // r1 = ptr_in+3 * COL/8+i +aor*COL/8;
            r3=((v8float *) in3)+i;
            data_buf1 = upd_w(data_buf1, 0, *r3++);
            data_buf1 = upd_w(data_buf1, 1, *r3);

            acc_0=fpmac   (acc_0,data_buf2,3,0x76543210,coeffs_rest,    0,0x00000000);  
            acc_0=fpmac   (acc_0,data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000);  

            // r1 = ptr_in+4 * COL/8+i +aor*COL/8;
            r4=((v8float *) in4)+i;
            data_buf1 = upd_w(data_buf1, 0, *r4++);
            data_buf1 = upd_w(data_buf1, 1, *r4);

            acc_0=fpmac   (acc_0,data_buf2,4,0x76543210,coeffs_rest,    0,0x00000000);  
            acc_0=fpmac   (acc_0,data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000); 

            // r1 = ptr_in+0 * COL/8+i +1+aor*COL/8;
            r0=((v8float *) in0)+i+1;
            data_buf1 = upd_w(data_buf1, 0, *r0++);
            data_buf1 = upd_w(data_buf1, 1, *r0);

            acc_0=fpmsc   (acc_0,data_buf2,2,0x76543210,coeffs,    0,0x00000000);      

            // r2 = ptr_in+2 * COL/8+i +1+aor*COL/8;
            r2=((v8float *) in2)+i+1;
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);            
                            
                    
            // window_writeincr(out, srs(acc_0,0));
            
            *ptr_out++ =  acc_0;       

        // }

    }
}

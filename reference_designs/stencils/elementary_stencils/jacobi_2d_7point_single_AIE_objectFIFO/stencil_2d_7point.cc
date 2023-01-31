// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET     
//
//===----------------------------------------------------------------------===//


// #include <adf.h>
#include "./include.h"
#include "stencil.h"
#define kernel_load 14
// typedef int int32;


//align to 16 bytes boundary, equivalent to "alignas(v4int32)"
void stencil_2d_7point(int32_t* restrict in1,int32_t* restrict in2,int32_t* restrict in3, int32_t* restrict out)
{
 // const int32_t *restrict w = weights;
    alignas(32) int32_t weights[8] = {-8,-8,-8,-8,-8,-8,-8,-8};
    alignas(32) int32_t weights_rest[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
   
    v8int32 coeffs         = *(v8int32*) weights;  //  8 x int32 = 256b W vector
    v8int32 coeffs_rest    = *(v8int32*) weights_rest;  //  8 x int32 = 256b W vector


    // v8int32 * restrict ptr_in = (v8int32 *) in->ptr;
    v8int32 * ptr_out = (v8int32 *) out;
    v8int32 * restrict r1=(v8int32 *) in1;
    v8int32 * restrict r2=(v8int32 *) in2;
    v8int32 * restrict r3=(v8int32 *) in3;
    v16int32 data_buf1 = null_v16int32();    
    v16int32 data_buf2 = null_v16int32();

    v8acc80 acc_0 = null_v8acc80();

    data_buf1 = upd_w(data_buf1, 0, *r1++);
    data_buf1 = upd_w(data_buf1, 1, *r1);

    data_buf2 = upd_w(data_buf2, 0, *r2++);
    data_buf2 = upd_w(data_buf2, 1, *r2);
    
        for (unsigned i = 0; i < COL/8; i++)
            chess_prepare_for_pipelining
                    chess_loop_range(1,)
            {
            acc_0=lmul8   (data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000);      
            acc_0=lmac8   (acc_0,data_buf2,1,0x76543210,coeffs_rest,    0,0x00000000); 

            r3=((v8int32 *) in3)+i;
            data_buf1 = upd_w(data_buf1, 0, *r3++);
            data_buf1 = upd_w(data_buf1, 1, *r3);
            
            acc_0=lmac8   (acc_0,data_buf2,3,0x76543210,coeffs_rest,    0,0x00000000);  
            acc_0=lmac8   (acc_0,data_buf1,2,0x76543210,coeffs_rest,    0,0x00000000);  

            r1=((v8int32 *) in1)+i+1;
            data_buf1 = upd_w(data_buf1, 0, *r1++);
            data_buf1 = upd_w(data_buf1, 1, *r1);
            acc_0=lmac8   (acc_0,data_buf2,1,0x76543210,coeffs_rest,    0,0x00000000);  
            acc_0=lmac8   (acc_0,data_buf2,3,0x76543210,coeffs_rest,    0,0x00000000);  
            acc_0=lmsc8   (acc_0,data_buf2,2,0x76543210,coeffs,    0,0x00000000);     


            r2=((v8int32 *) in2)+i+1;
            data_buf2 = upd_w(data_buf2, 0, *r2++);
            data_buf2 = upd_w(data_buf2, 1, *r2);      
            // window_writeincr(out, srs(acc_0,0));
            *ptr_out++ =  srs(acc_0,0);   
        // }

    }
}

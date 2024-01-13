//===- kernel.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

//#define __AIENGINE__ 1
#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1
// #include <aie_api/aie.hpp>
const int32_t MAX=255;
extern "C" {
enum region{top,middle,bottom};


    void conv2dk3(uint8_t *line0, uint8_t *line1,uint8_t *line2,int8_t *wts, uint8_t *output,  const int32_t  input_width,  const int32_t  input_channels, const int32_t  output_channels,
                        const int32_t  kernel_width,  const int32_t  kernel_height,  const int32_t  check, const int scale, const int channel_offset)                   
    {
        event0();

        int x, ki, ic, oc, ic8, oc8;
        int32_t sum;
        int sum_srs;
        int wts_indx_0=0,wts_indx_1=0,wts_indx_2=0;
        int in_indx_0=0;
        // for (oc = (0+channel_offset)/8; oc < (output_channels+channel_offset)/8; oc++) {  
        for (oc = 0; oc < output_channels/8; oc++) {
            int oc_ofst = oc + (channel_offset/8);
        for (oc8 = 0; oc8 < 8; oc8++) {  

            // left border
            sum = 0;   
            sum_srs=0;          
            for (ic = 0; ic < input_channels/8; ic++) {
            for (ic8 = 0; ic8 < 8; ic8++) {
                for (ki = 1; ki < kernel_width; ki++) {

                    // replicate 1 border pixel on the left
                    // wts_indx_0=0*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                    // wts_indx_1=1*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                    // wts_indx_2=2*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc; 
                    int wts_indx_0 = (0*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                    int wts_indx_1 = (1*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                    int wts_indx_2 = (2*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                    
                    if(ki==0) {                          
                        // in_indx_0=0+ki+input_width*ic;
                        in_indx_0 = (0+ki)*8 + ((ic*input_width*8)+ic8);
                    } else {
                        // in_indx_0=0+ki-1+input_width*ic;
                        in_indx_0 = (0+ki-1)*8 + ((ic*input_width*8)+ic8);
                    }
        
                    if(check != top)
                        sum += line0[in_indx_0] * wts[wts_indx_0];
                    sum += line1[in_indx_0] * wts[wts_indx_1];
                    if(check != bottom)
                        sum += line2[in_indx_0] * wts[wts_indx_2];            
                }
            }
            }
            // output[oc * (input_width) +  0] = sum;
            sum_srs=(sum+(1<<(scale-1))) >> scale;
            sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
            // output[oc * input_width + 0] = sum_srs;
            output[(oc*input_width*8) + oc8] = sum_srs;
                        
            // right border
            sum = 0;
            sum_srs=0;
            for (ic = 0; ic < input_channels/8; ic++) {
            for (ic8 = 0; ic8 < 8; ic8++) {
                for (ki = 0; ki < kernel_width-1; ki++) {
                    // wts_indx_0=0*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                    // wts_indx_1=1*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                    // wts_indx_2=2*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                    int wts_indx_0 = (0*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                    int wts_indx_1 = (1*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                    int wts_indx_2 = (2*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                                            
                    if(ki!=2) {
                        // in_indx_0=input_width-2+ki+input_width*ic;
                        in_indx_0 = (input_width-2+ki)*8 + ((ic*input_width*8)+ic8);
                    } else {  // replicate 1 border pixel on the right
                        // in_indx_0=input_width-2+ki-1+input_width*ic;
                        in_indx_0 = (input_width-2+ki-1)*8 + ((ic*input_width*8)+ic8);
                    }
                    if(check != top)
                        sum += line0[in_indx_0] * wts[wts_indx_0];
                    sum += line1[in_indx_0] * wts[wts_indx_1];
                    if(check != bottom)
                        sum += line2[in_indx_0] * wts[wts_indx_2];            
                }
            }                
            }
            sum_srs=(sum+(1<<(scale-1))) >> scale;
            sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
            // output[oc * input_width + input_width-1] = sum_srs;
            output[(oc*input_width*8) + (input_width-1)*8 + oc8] = sum_srs;
            // output[oc * (input_width) +  input_width-1] = sum;

            for (x = 1; x < input_width-1; x++) 
            { // col of output image
                sum = 0;
                sum_srs=0;
                for (ic = 0; ic < input_channels/8; ic++) {
                for (ic8 = 0; ic8 < 8; ic8++) {
                        for (ki = 0; ki < kernel_width; ki++) {
                            // wts format - orig is oc,ic,ky,kx, reformat is oc,ic,k0..k8,ic8,oc8

                            // int wts_indx_0=0*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                            // int wts_indx_1=1*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                            // int wts_indx_2=2*3 + ki + 3*kernel_width*ic + 3*kernel_width*input_channels*oc;
                            int wts_indx_0 = (0*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                            int wts_indx_1 = (1*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                            int wts_indx_2 = (2*3*64) + (ki*64) + (ic*3*kernel_width*64) + (ic8*8) + (oc_ofst*(input_channels/8)*3*kernel_width*64) + oc8;
                            
                            // int in_indx_0=x-1+ki+input_width*ic;
                            int in_indx_0 = (x-1+ki)*8 + ((ic*input_width*8)+ic8);
                
                            if(check != top)
                                sum += line0[in_indx_0] * wts[wts_indx_0];
                            sum += line1[in_indx_0] * wts[wts_indx_1];
                            if(check != bottom)
                                sum += line2[in_indx_0] * wts[wts_indx_2];            
                        }
                }
                } 
                sum_srs=(sum+(1<<(scale-1))) >> scale;
                sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
                output[(oc*input_width*8) + x*8 + oc8] = sum_srs;
                // output[oc * (input_width) +  x] = sum;
            }

        }
        }

        event1();
    }

}




//===- conv2dk1_skip.cc -------------------------------------------------*- C++
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

#include <aie_api/aie.hpp>


#ifdef SCALAR

const int32_t MIN=128;
const int32_t MAX=127;
const int32_t UMAX=255;
void conv2dk1_skip_init_i8_scalar(uint8_t *input0,uint8_t *input1,  int8_t *kernels, uint8_t *output, int8_t *skip, const int32_t  input_width, const int32_t  input_channels,const int32_t  output_channels,const int32_t  input_channels_skip,const int scale,const int skip_scale ,const int scale_skip_conv )                    

    {
        event0();

        int x,ic,ic2,ic3,oc,oc8,ic8,ic8b,ic8c;

        const int scaleT = scale;
        const int skip_scaleT = skip_scale;
        const int skip_scaleT_conv = scale_skip_conv;
        const int wts_offset=output_channels*input_channels;

        // const int scaleT = 10;
        // const int skip_scaleT = 0;

        for (oc = 0; oc < output_channels/8; oc++) {
            for (oc8 = 0; oc8 < 8; oc8++) {
                for (x = 0; x < input_width; x++) { // col of output image
                    int sum = 0;
                    int sum_srs=0;
                    int sum_skip_conv = 0;
                    int sum_skip_conv_srs=0;
                    int64_t skip_sum=0;
                    int skip_sum_srs_final=0;
                    int skip_sum_srs_final_out=0;
                    int skip_temp=0;
                    for (ic = 0; ic < input_channels/16; ic++) {
                        for (ic8 = 0; ic8 < 8; ic8++) {
                            // int val = input0[ic * input_width + x];
                            int val = input0[(ic*input_width*8) + (x*8) + ic8];
                            // int k = kernels[oc * input_channels + ic];
                            int k = kernels[(oc*(input_channels/8)*64) + (ic*64) + (ic8*8) + oc8];
                            sum += val * k;
                        }
                    }
                    for (ic2 = 0; ic2 < input_channels/16; ic2++) {                
                        for (ic8b = 0; ic8b < 8; ic8b++) {
                            // int val2 = input1[ic2 * input_width + x];
                            int val2 = input1[(ic2*input_width*8) + (x*8) + ic8b]; // TODO ic2 should be shifted?
                            // int k2 = kernels[oc * input_channels + ic2];
                            int k2 = kernels[(oc*(input_channels/8)*64) + ((ic2+(input_channels/16))*64) + (ic8b*8) + oc8];
                            sum += val2 * k2;
                        }
                    }
                    // scale for convolution
                    sum_srs=(sum+(1<<(scaleT-1))) >> scaleT;
                    sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < -MIN) ? -MIN : sum_srs; //clip
                    // sum_srs = (sum_srs > UMAX) ? UMAX : (sum_srs < 0) ? 0 : sum_srs; //clip
//  ********************************************************************************************************************
                    //skip convolution
                    for (ic3 = 0; ic3 < input_channels_skip/8; ic3++) {
                        for (ic8c = 0; ic8c < 8; ic8c++) {
                            int val3 = skip[(ic3*input_width*8) + (x*8) + ic8c];
                            int k3 = kernels[(oc*(input_channels_skip/8)*64) + (ic3*64) + (ic8c*8) + oc8+wts_offset];
                            sum_skip_conv += val3 * k3;
                        }
                    }
                    sum_skip_conv_srs =(sum_skip_conv+(1<<(skip_scaleT_conv-1))) >> skip_scaleT_conv;
                    sum_skip_conv_srs = (sum_skip_conv_srs > MAX) ? MAX : (sum_skip_conv_srs < -MIN) ? -MIN : sum_skip_conv_srs;                    
//  ********************************************************************************************************************
                    // scale for residual
                    // skip_temp=skip[oc * input_width + x];
                    // skip_temp=skip[(oc*input_width*8) + (x*8) + oc8] ;
                    skip_temp=sum_skip_conv_srs;
                    skip_sum= sum_srs+ skip_temp;
                    skip_sum_srs_final= (skip_sum+(1<<(skip_scaleT-1))) >> skip_scaleT;
                    skip_sum_srs_final_out = (skip_sum_srs_final > UMAX) ? UMAX : (skip_sum_srs_final < 0) ? 0 : skip_sum_srs_final; //clip
                                    
                    // output[oc * input_width + x] = skip_sum_srs_final_out;
                    output[(oc*input_width*8) + (x*8) + oc8] = skip_sum_srs_final_out;
            
                    // output[oc * input_width + x] = sum;
                    // output[oc * input_width + x] = sum+skip[oc * input_width + x];
                }
            }
        }

        // for (oc = 0; oc < output_channels; ++oc) {
        //         for (x = 0; x < input_width; ++x) { 
        //             output[oc * input_width + x]=skip[oc * input_width + x];}
        // }

        event1();
}


#else // Vector


#endif

extern "C" {

#ifdef SCALAR

void conv2dk1_skip_init_i8(uint8_t *input0,uint8_t *input1,  int8_t *kernels, uint8_t *output, int8_t *skip, const int32_t  input_width, const int32_t  input_channels,const int32_t  output_channels,const int32_t  input_channels_skip,const int scale,const int skip_scale ,const int scale_skip_conv )                    
{
    conv2dk1_skip_init_i8_scalar(input0, input1,  kernels, output,skip,  input_width, input_channels, output_channels,input_channels_skip,scale,skip_scale ,scale_skip_conv ) ;

}


#else // Vector

#endif


} // extern "C"
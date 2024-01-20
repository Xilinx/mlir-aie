//===- pass_through.cc -------------------------------------------------*- C++
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

// #ifdef SCALAR

// NOTE: Assumes input_channels >= 16
void pass_through_scalar(uint8_t *input0,uint8_t *input1,  uint8_t *output, const int32_t  input_width, const int32_t  input_channels,const int32_t  output_channels )                    
{
    event0();

    int x,ic,ic2,oc,oc8,ic8,ic8b;
    int oc2, oc8b;

    for (oc = 0; oc < output_channels/8; oc++) 
    {
        if(oc>=output_channels/16)
        {
            for (oc8b = 0; oc8b < 8; oc8b++) {
                    for (x = 0; x < input_width; x++) { // col of output image
                        int val2 = input1[((oc-(output_channels/16))*input_width*8) + (x*8) + oc8b];
                        output[(oc*input_width*8) + (x*8) + oc8b] = val2;                      
                        }
                    }
        }
        else
        {
            for (oc8 = 0; oc8 < 8; oc8++) {
                for (x = 0; x < input_width; x++) { // col of output image
                    int val = input0[(oc*input_width*8) + (x*8) + oc8];
                    output[(oc*input_width*8) + (x*8) + oc8] = val;
                }
            }
        }
        
    event1();
    }
}

// #else // Vector

// #endif
extern "C" {

// #ifdef SCALAR
void pass_through(uint8_t *input0,uint8_t *input1,  uint8_t *output, const int32_t  input_width, const int32_t  input_channels,const int32_t  output_channels )  
{
    pass_through_scalar(input0, input1, output, input_width, input_channels, output_channels);
}

// #else // Vector

// #endif

} // extern "C"
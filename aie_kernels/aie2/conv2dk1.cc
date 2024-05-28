//===- conv2dk1.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

//*****************************************************************************
// conv2d 1x1 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************
void conv2dk1_i8_scalar(int8_t *input, int8_t *kernels, uint8_t *output,
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels, const int scale) {
  event0();

  int x, ic, oc, ic8, oc8;
  // scale=-17;
  for (oc = 0; oc < output_channels / 8; oc++) {
    for (x = 0; x < input_width; x++) { // col of output image
      for (oc8 = 0; oc8 < 8; oc8++) {
        int sum = 0;
        int sum_srs = 0;

        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            int val = input[(ic * input_width * 8) + (x * 8) + ic8];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }

        // sum_srs=sum>>scale;
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > UINT8_MAX) ? UINT8_MAX : (sum_srs < 0) ? 0 : sum_srs;
        // sum_srs = input[(oc*input_width*8) + (x*8) + oc8];
        output[(oc * input_width * 8) + (x * 8) + oc8] = sum_srs;
      }
    }
  }

  event1();
}

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

void conv2dk1_relu_i8_ui8(int8_t *input, int8_t *kernels, uint8_t *output,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale) {
  
  conv2dk1_i8_scalar(input, kernels, output, input_width, input_channels,
                     output_channels, scale);

}

} // extern "C"
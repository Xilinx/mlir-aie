//===- conv2dk1_skip_init.cc -------------------------------------------------*-
// C++
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

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#ifdef SCALAR

const int32_t MIN = 128;
const int32_t MAX = 127;
const int32_t UMAX = 255;

//*****************************************************************************
// conv2d 1x1 skip - scalar
// act: uint8, wts: int8, skip: int8, out: int8
//*****************************************************************************
void conv2dk1_skip_ui8_i8_i8_scalar(
    uint8_t *input0, int8_t *kernels, int8_t *output, int8_t *skip, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int scale, const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  // const int scaleT = 10;
  // const int skip_scaleT = 0;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int64_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        int skip_temp = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            // int val = input0[ic * input_width + x];
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            // int k = kernels[oc * input_channels + ic];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // scale for convolution
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip
        // //clip

        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs + skip_temp;

        skip_sum_srs_final =
            (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
                                 : (skip_sum_srs_final < -MIN)
                                     ? -MIN
                                     : skip_sum_srs_final; // clip

        // output[oc * input_width + x] = skip_sum_srs_final_out;
        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;

        // output[oc * input_width + x] = sum;
        // output[oc * input_width + x] = sum+skip[oc * input_width + x];
      }
    }
  }

  event1();
}

void conv2dk1_skip_ui8_ui8_i8_scalar(
    uint8_t *input0, int8_t *kernels, int8_t *output, uint8_t *skip, 
    const int32_t input_width, const int32_t input_channels, const int32_t output_channels,
    const int scale, const int skip_scale) {
  event0();

  int x, ic, ic2, oc, oc8, ic8, ic8b;

  const int scaleT = scale;
  const int skip_scaleT = skip_scale;
  // const int scaleT = 10;
  // const int skip_scaleT = 0;

  for (oc = 0; oc < output_channels / 8; oc++) {
    for (oc8 = 0; oc8 < 8; oc8++) {
      for (x = 0; x < input_width; x++) { // col of output image
        int sum = 0;
        int sum_srs = 0;
        int64_t skip_sum = 0;
        int skip_sum_srs_final = 0;
        int skip_sum_srs_final_out = 0;
        int skip_temp = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            // int val = input0[ic * input_width + x];
            int val = input0[(ic * input_width * 8) + (x * 8) + ic8];
            // int k = kernels[oc * input_channels + ic];
            int k = kernels[(oc * (input_channels / 8) * 64) + (ic * 64) +
                            (ic8 * 8) + oc8];
            sum += val * k;
          }
        }
        // scale for convolution
        sum_srs = (sum + (1 << (scaleT - 1))) >> scaleT;
        sum_srs = (sum_srs > MAX)    ? MAX
                  : (sum_srs < -MIN) ? -MIN
                                     : sum_srs; // clip
        // //clip

        skip_temp = skip[(oc * input_width * 8) + (x * 8) + oc8];
        skip_sum = sum_srs + skip_temp;

        skip_sum_srs_final =
            (skip_sum + (1 << (skip_scaleT - 1))) >> skip_scaleT;
        skip_sum_srs_final_out = (skip_sum_srs_final > MAX) ? MAX
                                 : (skip_sum_srs_final < -MIN)
                                     ? -MIN
                                     : skip_sum_srs_final; // clip

        // output[oc * input_width + x] = skip_sum_srs_final_out;
        output[(oc * input_width * 8) + (x * 8) + oc8] = skip_sum_srs_final_out;

        // output[oc * input_width + x] = sum;
        // output[oc * input_width + x] = sum+skip[oc * input_width + x];
      }
    }
  }

  event1();
}

#else // Vector


#endif // Vector
//*****************************************************************************
// conv2d 1x1 skip wrappers
//*****************************************************************************
extern "C" {

#ifdef SCALAR

#ifdef UNSIGNED_SKIP

void conv2dk1_skip_ui8_ui8_i8(uint8_t *input0,int8_t *kernels,
                       int8_t *output, uint8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_ui8_i8_scalar(input0,  kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#else

void conv2dk1_skip_ui8_i8_i8(uint8_t *input0,int8_t *kernels,
                       int8_t *output, int8_t *skip,
                       const int32_t input_width, const int32_t input_channels,
                       const int32_t output_channels, const int scale,
                       const int skip_scale) {
  conv2dk1_skip_ui8_i8_i8_scalar(input0,  kernels, output, skip, input_width,
                           input_channels, output_channels, scale, skip_scale);
}

#endif // UNSIGNED_SKIP

#else // Vector


#endif // Vector

} // extern "C"
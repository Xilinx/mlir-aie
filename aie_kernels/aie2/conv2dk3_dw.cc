//===- conv2dk3.cc -------------------------------------------------*- C++
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

enum region { top, middle, bottom };

#ifdef SCALAR

const int32_t MAX = 255;

//*****************************************************************************
// conv2d 3x3 - scalar
// act: uint8, wts: int8, out: uint8
//*****************************************************************************
  #ifdef STRIDE2

  void conv2dk3_stride2_ui8_scalar(uint8_t *line0, uint8_t *line1, uint8_t *line2,
                          int8_t *wts, uint8_t *output,
                          const int32_t input_width,
                          const int32_t input_channels,
                          const int32_t output_channels,
                          const int32_t kernel_width,
                          const int32_t kernel_height, const int32_t check,
                          const int scale, const int channel_offset) 
    {
event0();

    int x, ki,c_div_8,c8;
    int32_t sum;
    int sum_srs;
    int wts_indx_0 = 0, wts_indx_1 = 0, wts_indx_2 = 0;
    int in_indx_0 = 0;
    int32_t output_width=input_width/2;
    // for (oc = (0+channel_offset)/8; oc < (output_channels+channel_offset)/8;
    // oc++) {
    int CHANNEL_REMAIN=output_channels / 8;
    int VECTOR_SIZE=8;
    for (c_div_8 = 0; c_div_8 < CHANNEL_REMAIN; c_div_8++) {
          for (c8 = 0; c8 < VECTOR_SIZE; c8++) {
            //left border
              sum = 0;
              sum_srs = 0;
              for (ki = 1; ki < kernel_width; ki++) {
                // wts format - orig is oc,ic,ky,kx, reformat is
                // oc,ic,k0..k8,ic8,oc8
                int wts_indx_0 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +0*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_1 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +1*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_2 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +2*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int in_indx_0=
                c_div_8 * input_width * VECTOR_SIZE
                +(0 + ki-1) * VECTOR_SIZE
                + c8;
                if (check != top)
                  sum += line0[in_indx_0] * wts[wts_indx_0];
                sum += line1[in_indx_0] * wts[wts_indx_1];
                if (check != bottom)
                  sum += line2[in_indx_0] * wts[wts_indx_2];
              }
            sum_srs = (sum + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
            output[(c_div_8 * output_width * VECTOR_SIZE) + c8] = sum_srs;

            for (x = 1; x < output_width; x++) { // middle part of row
              sum = 0;
              sum_srs = 0;
              for (ki = 0; ki < kernel_width; ki++) {
                // wts format - orig is oc,ic,ky,kx, reformat is
                // oc,ic,k0..k8,ic8,oc8
                int wts_indx_0 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +0*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_1 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +1*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_2 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +2*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int in_indx_0=
                c_div_8 * input_width * VECTOR_SIZE
                +(2*x-1 + ki) * VECTOR_SIZE
                + c8;
                if (check != top)
                  sum += line0[in_indx_0] * wts[wts_indx_0];
                sum += line1[in_indx_0] * wts[wts_indx_1];
                if (check != bottom)
                  sum += line2[in_indx_0] * wts[wts_indx_2];
              }
            sum_srs = (sum + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
            output[(c_div_8 * output_width * VECTOR_SIZE) + x * VECTOR_SIZE + c8] = sum_srs;
            }
          
          }
    }

    event1();
  }
  #else
  void conv2dk3_ui8_scalar(uint8_t *line0, uint8_t *line1, uint8_t *line2,
                          int8_t *wts, uint8_t *output,
                          const int32_t input_width,
                          const int32_t input_channels,
                          const int32_t output_channels,
                          const int32_t kernel_width,
                          const int32_t kernel_height, const int32_t check,
                          const int scale, const int channel_offset) 
    {
    event0();

    int x, ki,c_div_8,c8;
    int32_t sum;
    int sum_srs;
    int wts_indx_0 = 0, wts_indx_1 = 0, wts_indx_2 = 0;
    int in_indx_0 = 0;
    // for (oc = (0+channel_offset)/8; oc < (output_channels+channel_offset)/8;
    // oc++) {
    int CHANNEL_REMAIN=output_channels / 8;
    int VECTOR_SIZE=8;
    for (c_div_8 = 0; c_div_8 < CHANNEL_REMAIN; c_div_8++) {
          for (c8 = 0; c8 < VECTOR_SIZE; c8++) {
            //left border
              sum = 0;
              sum_srs = 0;
              for (ki = 1; ki < kernel_width; ki++) {
                // wts format - orig is oc,ic,ky,kx, reformat is
                // oc,ic,k0..k8,ic8,oc8
                int wts_indx_0 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +0*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_1 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +1*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_2 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +2*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int in_indx_0=
                c_div_8 * input_width * VECTOR_SIZE
                +(0 + ki-1) * VECTOR_SIZE
                + c8;
                if (check != top)
                  sum += line0[in_indx_0] * wts[wts_indx_0];
                sum += line1[in_indx_0] * wts[wts_indx_1];
                if (check != bottom)
                  sum += line2[in_indx_0] * wts[wts_indx_2];
              }
            sum_srs = (sum + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
            output[(c_div_8 * input_width * VECTOR_SIZE) + c8] = sum_srs;

            //right border
              sum = 0;
              sum_srs = 0;
              for (ki = 0; ki < kernel_width-1; ki++) {
                // wts format - orig is oc,ic,ky,kx, reformat is
                // oc,ic,k0..k8,ic8,oc8
                int wts_indx_0 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +0*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_1 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +1*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_2 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +2*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                  
                int in_indx_0=
                c_div_8 * input_width * VECTOR_SIZE
                +(input_width-2 + ki) * VECTOR_SIZE
                + c8;
                if (check != top)
                  sum += line0[in_indx_0] * wts[wts_indx_0];
                sum += line1[in_indx_0] * wts[wts_indx_1];
                if (check != bottom)
                  sum += line2[in_indx_0] * wts[wts_indx_2];
              }
            sum_srs = (sum + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
            output[(c_div_8 * input_width * VECTOR_SIZE) + (input_width-1) * VECTOR_SIZE + c8] = sum_srs;

            for (x = 1; x < input_width - 1; x++) { // middle part of row
              sum = 0;
              sum_srs = 0;
              for (ki = 0; ki < kernel_width; ki++) {
                // wts format - orig is oc,ic,ky,kx, reformat is
                // oc,ic,k0..k8,ic8,oc8
                int wts_indx_0 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +0*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_1 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +1*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int wts_indx_2 =
                    +3*3*VECTOR_SIZE*c_div_8
                    +2*3*VECTOR_SIZE
                    +ki*VECTOR_SIZE
                    +c8;
                int in_indx_0=
                c_div_8 * input_width * VECTOR_SIZE
                +(x-1 + ki) * VECTOR_SIZE
                + c8;
                if (check != top)
                  sum += line0[in_indx_0] * wts[wts_indx_0];
                sum += line1[in_indx_0] * wts[wts_indx_1];
                if (check != bottom)
                  sum += line2[in_indx_0] * wts[wts_indx_2];
              }
            sum_srs = (sum + (1 << (scale - 1))) >> scale;
            sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
            output[(c_div_8 * input_width * VECTOR_SIZE) + x * VECTOR_SIZE + c8] = sum_srs;
            // output[oc * (input_width) +  x] = sum;
            }
          
          }
    }

    event1();
  }
  #endif //STRIDE
#else // Vector


#endif // Vector

extern "C" {

#ifdef SCALAR

  #ifdef STRIDE2
  void conv2dk3_ui8(uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
                    uint8_t *output, const int32_t input_width,
                    const int32_t input_channels, const int32_t output_channels,
                    const int32_t kernel_width, const int32_t kernel_height,
                    const int32_t check, const int scale,
                    const int channel_offset) {
    conv2dk3_stride2_ui8_scalar(line0, line1, line2, wts, output, input_width,
                        input_channels, output_channels, kernel_width,
                        kernel_height, check, scale, channel_offset);
  }
  #else
  void conv2dk3_ui8(uint8_t *line0, uint8_t *line1, uint8_t *line2, int8_t *wts,
                    uint8_t *output, const int32_t input_width,
                    const int32_t input_channels, const int32_t output_channels,
                    const int32_t kernel_width, const int32_t kernel_height,
                    const int32_t check, const int scale,
                    const int channel_offset) {
    conv2dk3_ui8_scalar(line0, line1, line2, wts, output, input_width,
                        input_channels, output_channels, kernel_width,
                        kernel_height, check, scale, channel_offset);
  }
  #endif 
#endif 
}
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
// #define __AIENGINE__ 2
#define NOCPP
// #define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

enum region { top, middle, bottom };

const int32_t MAX = 255;

//*****************************************************************************
// conv2d 3x3 - scalar
// act: int8, wts: int8, out: uint8
//*****************************************************************************

 void conv2dk3_i8_stride2_scalar(int8_t *line0, int8_t *line1, int8_t *line2,
                                int8_t *wts, uint8_t *output, const int32_t input_width,
                                const int32_t input_channels,
                                const int32_t output_channels,
                                const int32_t kernel_width, const int32_t kernel_height,
                                const int32_t check, const int scale,
                                const int channel_offset) {
  event0();

  int x, ki, ic, oc, ic8, oc8;
  int32_t sum;
  int sum_srs;
  int wts_indx_0 = 0, wts_indx_1 = 0, wts_indx_2 = 0;
  int in_indx_0 = 0;
  int output_width = input_width / 2; // Stride 2 reduces output width by half

  for (oc = 0; oc < output_channels / 8; oc++) {
    int oc_ofst = oc + (channel_offset / 8);
    for (oc8 = 0; oc8 < 8; oc8++) {

      // left border
      sum = 0;
      sum_srs = 0;
      for (ic = 0; ic < input_channels / 8; ic++) {
        for (ic8 = 0; ic8 < 8; ic8++) {
          for (ki = 1; ki < kernel_width; ki++) {
            int wts_indx_0 = (0 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) + (ic8 * 8) +
                             (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;
            int wts_indx_1 = (1 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) + (ic8 * 8) +
                             (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;
            int wts_indx_2 = (2 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) + (ic8 * 8) +
                             (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;

            in_indx_0 = (0 + ki - 1) * 8 + ((ic * input_width * 8) + ic8);

            if (check != top)
              sum += line0[in_indx_0] * wts[wts_indx_0];
            sum += line1[in_indx_0] * wts[wts_indx_1];
            if (check != bottom)
              sum += line2[in_indx_0] * wts[wts_indx_2];
          }
        }
      }
      // sum_srs = (sum + (1 << (scale - 1))) >> scale;
      sum_srs = (((sum) + (1 << (scale - 1)) - 1 + (((sum) >> scale) & 1)) >> scale);
      sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
      output[(oc * output_width * 8) + oc8] = sum_srs;

      for (x = 1; x < output_width; x++) { // stride 2 means we skip one input column
        sum = 0;
        sum_srs = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            for (ki = 0; ki < kernel_width; ki++) {
              int wts_indx_0 = (0 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) + (ic8 * 8) +
                               (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;
              int wts_indx_1 = (1 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) + (ic8 * 8) +
                               (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;
              int wts_indx_2 = (2 * 3 * 64) + (ki * 64) + (ic * 3 * kernel_width * 64) + (ic8 * 8) +
                               (oc_ofst * (input_channels / 8) * 3 * kernel_width * 64) + oc8;

              int in_indx_0 = (2 * x - 1 + ki) * 8 + ((ic * input_width * 8) + ic8);

              if (check != top)
                sum += line0[in_indx_0] * wts[wts_indx_0];
              sum += line1[in_indx_0] * wts[wts_indx_1];
              if (check != bottom)
                sum += line2[in_indx_0] * wts[wts_indx_2];
            }
          }
        }
        // sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (((sum) + (1 << (scale - 1)) - 1 + (((sum) >> scale) & 1)) >> scale);
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(oc * output_width * 8) + x * 8 + oc8] = sum_srs;
      }
    }
  }

  event1();
}

 

extern "C" {


 void conv2dk3_stride2_i8(int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts,
                 uint8_t *output, const int32_t input_width,
                 const int32_t input_channels, const int32_t output_channels,
                 const int32_t kernel_width, const int32_t kernel_height,
                 const int32_t check, const int scale,
                 const int channel_offset) {
  conv2dk3_i8_stride2_scalar(line0, line1, line2, wts, output, input_width,
                     input_channels, output_channels, kernel_width,
                     kernel_height, check, scale, channel_offset);
}


}
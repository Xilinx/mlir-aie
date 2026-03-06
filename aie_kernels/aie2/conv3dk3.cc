//===- conv3dk3.cc ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

enum region { top_plane, middle_plane, bottom_plane };

#ifdef SCALAR

const int32_t MAX = 255;

extern "C" {

//*****************************************************************************
// conv3d 3x3x3 - scalar
// act: uint8, wts: int8, out: uint8
//
// Processes a single depth plane of output, using three input depth planes
// (z-1, z, z+1) to apply the 3x3x3 convolution kernel.
//
// Input layout: HxWxC (height, width, channels)
// Weight layout: {O/8}{I/8}KDHW{I8}{O8} where KD=3, KH=3, KW=3
// Output layout: HxWxC
//
// For border handling:
//   - check==top_plane: skip z-1 plane (use only z and z+1)
//   - check==bottom_plane: skip z+1 plane (use only z-1 and z)
//   - check==middle_plane: use all three planes
//*****************************************************************************
void conv3dk3_ui8_scalar(uint8_t *plane0, uint8_t *plane1, uint8_t *plane2,
                         int8_t *wts, uint8_t *output,
                         const int32_t input_width, const int32_t input_height,
                         const int32_t input_channels,
                         const int32_t output_channels,
                         const int32_t kernel_width, const int32_t kernel_height,
                         const int32_t kernel_depth, const int32_t check,
                         const int scale, const int channel_offset) {
  event0();

  int x, y, kh, kw, kd, ic, oc, ic8, oc8;
  int32_t sum;
  int sum_srs;
  const int plane_size = input_height * input_width * 8;

  // Loop over output channel groups (groups of 8)
  for (oc = 0; oc < output_channels / 8; oc++) {
    int oc_ofst = oc + (channel_offset / 8);

    for (oc8 = 0; oc8 < 8; oc8++) {
      // Process each row of the output plane
      for (y = 0; y < input_height; y++) {
        // Left border (x=0)
        sum = 0;
        sum_srs = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            for (kd = 0; kd < kernel_depth; kd++) {
              for (kh = 0; kh < kernel_height; kh++) {
                for (kw = 1; kw < kernel_width; kw++) {
                  // Weight index for 3D: {O/8}{I/8}KDHW{I8}{O8}
                  // Formula: (kd*KH*KW*64) + (kh*KW*64) + (kw*64) +
                  //          (ic*KD*KH*KW*64) + (ic8*8) +
                  //          (oc_ofst*(IC/8)*KD*KH*KW*64) + oc8
                  int wts_indx = (kd * 3 * 3 * 64) + (kh * 3 * 64) +
                                 (kw * 64) +
                                 (ic * 3 * 3 * 3 * 64) + (ic8 * 8) +
                                 (oc_ofst * (input_channels / 8) * 3 * 3 * 3 *
                                  64) +
                                 oc8;

                  // Input index (with left border replication)
                  int y_pos = (y - 1 + kh < 0)
                                  ? 0
                                  : ((y - 1 + kh >= input_height)
                                         ? input_height - 1
                                         : y - 1 + kh);
                  int x_pos = (kw == 0) ? 0 : kw - 1;
                  int in_indx =
                      (y_pos * input_width + x_pos) * 8 + (ic * plane_size) + ic8;

                  // Accumulate from the three depth planes
                  if (kd == 0 && check != top_plane)
                    sum += plane0[in_indx] * wts[wts_indx];
                  else if (kd == 1)
                    sum += plane1[in_indx] * wts[wts_indx];
                  else if (kd == 2 && check != bottom_plane)
                    sum += plane2[in_indx] * wts[wts_indx];
                }
              }
            }
          }
        }
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(oc * input_height * input_width * 8) + (y * input_width * 8) +
               oc8] = sum_srs;

        // Right border (x=input_width-1)
        sum = 0;
        sum_srs = 0;
        for (ic = 0; ic < input_channels / 8; ic++) {
          for (ic8 = 0; ic8 < 8; ic8++) {
            for (kd = 0; kd < kernel_depth; kd++) {
              for (kh = 0; kh < kernel_height; kh++) {
                for (kw = 0; kw < kernel_width - 1; kw++) {
                  int wts_indx = (kd * 3 * 3 * 64) + (kh * 3 * 64) +
                                 (kw * 64) +
                                 (ic * 3 * 3 * 3 * 64) + (ic8 * 8) +
                                 (oc_ofst * (input_channels / 8) * 3 * 3 * 3 *
                                  64) +
                                 oc8;

                  int y_pos = (y - 1 + kh < 0)
                                  ? 0
                                  : ((y - 1 + kh >= input_height)
                                         ? input_height - 1
                                         : y - 1 + kh);
                  int x_pos = (kw == 2) ? input_width - 1 : input_width - 2 + kw;
                  int in_indx =
                      (y_pos * input_width + x_pos) * 8 + (ic * plane_size) + ic8;

                  if (kd == 0 && check != top_plane)
                    sum += plane0[in_indx] * wts[wts_indx];
                  else if (kd == 1)
                    sum += plane1[in_indx] * wts[wts_indx];
                  else if (kd == 2 && check != bottom_plane)
                    sum += plane2[in_indx] * wts[wts_indx];
                }
              }
            }
          }
        }
        sum_srs = (sum + (1 << (scale - 1))) >> scale;
        sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
        output[(oc * input_height * input_width * 8) +
               (y * input_width * 8) + (input_width - 1) * 8 + oc8] = sum_srs;

        // Middle columns (x=1 to input_width-2)
        for (x = 1; x < input_width - 1; x++) {
          sum = 0;
          sum_srs = 0;
          for (ic = 0; ic < input_channels / 8; ic++) {
            for (ic8 = 0; ic8 < 8; ic8++) {
              for (kd = 0; kd < kernel_depth; kd++) {
                for (kh = 0; kh < kernel_height; kh++) {
                  for (kw = 0; kw < kernel_width; kw++) {
                    // Weight index for 3D convolution
                    int wts_indx = (kd * 3 * 3 * 64) + (kh * 3 * 64) +
                                   (kw * 64) +
                                   (ic * 3 * 3 * 3 * 64) + (ic8 * 8) +
                                   (oc_ofst * (input_channels / 8) * 3 * 3 * 3 *
                                    64) +
                                   oc8;

                    // Input index with height border handling
                    int y_pos = (y - 1 + kh < 0)
                                    ? 0
                                    : ((y - 1 + kh >= input_height)
                                           ? input_height - 1
                                           : y - 1 + kh);
                    int x_pos = x - 1 + kw;
                    int in_indx = (y_pos * input_width + x_pos) * 8 +
                                  (ic * plane_size) + ic8;

                    // Accumulate from three depth planes
                    if (kd == 0 && check != top_plane)
                      sum += plane0[in_indx] * wts[wts_indx];
                    else if (kd == 1)
                      sum += plane1[in_indx] * wts[wts_indx];
                    else if (kd == 2 && check != bottom_plane)
                      sum += plane2[in_indx] * wts[wts_indx];
                  }
                }
              }
            }
          }
          sum_srs = (sum + (1 << (scale - 1))) >> scale;
          sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;
          output[(oc * input_height * input_width * 8) +
                 (y * input_width * 8) + x * 8 + oc8] = sum_srs;
        }
      }
    }
  }

  event1();
}

} // extern "C"

#else // Vectorized implementation (placeholder for Phase 2)

extern "C" {

//*****************************************************************************
// conv3d 3x3x3 - vectorized (AIE intrinsics)
// TODO: Implement vectorized version in Phase 2
//*****************************************************************************
void conv3dk3_ui8(uint8_t *plane0, uint8_t *plane1, uint8_t *plane2,
                  int8_t *wts, uint8_t *output, const int32_t input_width,
                  const int32_t input_height, const int32_t input_channels,
                  const int32_t output_channels, const int32_t kernel_width,
                  const int32_t kernel_height, const int32_t kernel_depth,
                  const int32_t check, const int scale,
                  const int channel_offset) {
  // Placeholder: For Phase 2 vectorized implementation
  // Will use aie::mmul<4, 8, 8, uint8, int8> and vector intrinsics
  event0();
  // TODO: Implement vectorized kernel using AIE MMUL intrinsics
  event1();
}

} // extern "C"

#endif // SCALAR

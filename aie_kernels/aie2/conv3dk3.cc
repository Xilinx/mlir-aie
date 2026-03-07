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
                  if (kd == 1)
                    sum += plane1[in_indx] * wts[wts_indx];
                  if (kd == 2 && check != bottom_plane)
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
                  if (kd == 1)
                    sum += plane1[in_indx] * wts[wts_indx];
                  if (kd == 2 && check != bottom_plane)
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
                    if (kd == 1)
                      sum += plane1[in_indx] * wts[wts_indx];
                    if (kd == 2 && check != bottom_plane)
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

#else // Vectorized implementation using AIE intrinsics

extern "C" {

//*****************************************************************************
// conv3d 3x3x3 - vectorized (AIE intrinsics)
// act: uint8, wts: int8, out: uint8
//
// Uses 4x8x8 matrix multiply (4 outputs x 8 channels x 8 weights)
//*****************************************************************************
void conv3dk3_ui8(uint8_t *plane0, uint8_t *plane1, uint8_t *plane2,
                  int8_t *wts, uint8_t *output, const int32_t input_width,
                  const int32_t input_height, const int32_t input_channels,
                  const int32_t output_channels, const int32_t kernel_width,
                  const int32_t kernel_height, const int32_t kernel_depth,
                  const int32_t check, const int scale,
                  const int channel_offset) {
  event0();

  constexpr int MMUL_M = 4;  // Process 4 output pixels at a time
  constexpr int MMUL_K = 8;  // Channel group size
  constexpr int MMUL_N = 8;  // Output channel group size
  constexpr int MMUL_MK = MMUL_M * MMUL_K;
  constexpr int MMUL_KN = MMUL_K * MMUL_N;
  constexpr int MMUL_MN = MMUL_M * MMUL_N;

  using MMUL = aie::mmul<MMUL_M, MMUL_K, MMUL_N, uint8, int8>;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::symmetric_inf);

  const int plane_size = input_height * input_width * 8;
  const int MAX = 255;

  // Process each output channel group
  for (int oc = 0; oc < output_channels / 8; oc++) {
    int oc_ofst = oc + (channel_offset / 8);

    // Process each row
    for (int y = 0; y < input_height; y++) {
      int y_start = (y - 1 < 0) ? 0 : y - 1;
      int y_end = (y + 1 >= input_height) ? input_height - 1 : y + 1;

      // Process pixels in groups of 4 for vectorization
      for (int x = 0; x < input_width; x += MMUL_M) {
        int x_valid = (x + MMUL_M <= input_width) ? MMUL_M : input_width - x;

        // Accumulator for this output group
        MMUL acc = aie::zeros<acc32, MMUL_MN>();

        // Accumulate over all input channel groups
        for (int ic = 0; ic < input_channels / 8; ic++) {
          // Accumulate over 3x3x3 kernel
          for (int kd = 0; kd < kernel_depth; kd++) {
            // Skip depth kernel positions based on check
            if (kd == 0 && check == top_plane) continue;
            if (kd == 2 && check == bottom_plane) continue;

            // Select the appropriate plane
            uint8_t *plane = (kd == 0) ? plane0 : (kd == 1) ? plane1 : plane2;

            for (int kh = 0; kh < kernel_height; kh++) {
              int y_pos = y_start + kh;
              if (y_pos < 0) y_pos = 0;
              if (y_pos >= input_height) y_pos = input_height - 1;

              for (int kw = 0; kw < kernel_width; kw++) {
                // Load weights for this kernel position
                int wts_idx = (kd * 3 * 3 * 64) + (kh * 3 * 64) + (kw * 64) +
                             (ic * 3 * 3 * 3 * 64) + (oc_ofst * (input_channels / 8) * 3 * 3 * 3 * 64);
                aie::vector<int8, MMUL_KN> w = aie::load_v<MMUL_KN>(wts + wts_idx);

                // Load activations for this group of pixels
                // Handle boundary conditions
                alignas(32) uint8 act_buf[MMUL_MK];
                for (int xx = 0; xx < x_valid && xx < MMUL_M; xx++) {
                  int x_pos = x + xx - 1 + kw;
                  if (x_pos < 0) x_pos = 0;
                  if (x_pos >= input_width) x_pos = input_width - 1;

                  int in_idx = (y_pos * input_width + x_pos) * 8 + (ic * plane_size);
                  for (int ch = 0; ch < 8; ch++) {
                    act_buf[xx * 8 + ch] = plane[in_idx + ch];
                  }
                }

                aie::vector<uint8, MMUL_MK> a = aie::load_v<MMUL_MK>(act_buf);
                acc.mac(a, w);
              }
            }
          }
        }

        // Convert accumulator to output with scaling
        aie::vector<uint8, MMUL_MN> o = acc.to_vector<uint8>(scale);

        // Store output, handling partial groups
        for (int xx = 0; xx < x_valid && x + xx < input_width; xx++) {
          for (int ch = 0; ch < 8; ch++) {
            int out_idx = (oc * input_height * input_width * 8) +
                         (y * input_width * 8) + ((x + xx) * 8) + ch;
            output[out_idx] = o[xx * 8 + ch];
          }
        }
      }
    }
  }

  event1();
}

} // extern "C"

#endif // SCALAR

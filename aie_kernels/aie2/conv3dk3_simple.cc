//===- conv3dk3_simple.cc ---------------------------------------*- C++ -*-===//
//
// Simplified conv3d kernel for debugging - processes ONE plane at a time
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"

extern "C" {

// Simplified 2D convolution on a single plane (no 3D yet)
// Just does 2D conv with 3x3 kernel on one HxW plane
void conv3dk3_simple(uint8_t *input_plane, int8_t *wts, uint8_t *output_plane,
                     const int32_t width, const int32_t height,
                     const int32_t in_channels, const int32_t out_channels) {
  event0();

  const int32_t MAX = 255;
  const int32_t scale = 10;  // Fixed scale
  const int plane_size = height * width * 8;

  // Simple 1x1 convolution for now (no spatial kernel)
  // Just process each pixel independently
  for (int oc = 0; oc < out_channels / 8; oc++) {
    for (int oc8 = 0; oc8 < 8; oc8++) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          int32_t sum = 0;

          for (int ic = 0; ic < in_channels / 8; ic++) {
            for (int ic8 = 0; ic8 < 8; ic8++) {
              int in_idx = (y * width + x) * 8 + (ic * plane_size) + ic8;
              int wt_idx = (oc * (in_channels / 8) * 64) + (ic * 64) + (ic8 * 8) + oc8;

              sum += input_plane[in_idx] * wts[wt_idx];
            }
          }

          // Scale and saturate
          int sum_srs = (sum + (1 << (scale - 1))) >> scale;
          sum_srs = (sum_srs > MAX) ? MAX : (sum_srs < 0) ? 0 : sum_srs;

          int out_idx = (oc * height * width * 8) + (y * width * 8) + x * 8 + oc8;
          output_plane[out_idx] = sum_srs;
        }
      }
    }
  }

  event1();
}

} // extern "C"

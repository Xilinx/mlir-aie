//===- yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.cc -----------------*- C++ -*-===//
//
// Scalar 3x3 stride-2 conv with OIYXI8O8 weight layout. Used by yolo26n
// m1/m3/m5/m7 (conv_stride blocks with 8-aligned in_c). Shape parameters
// are runtime args, so a single .o serves all four blocks.
//
// Numerics mirror kernels/yolo_m0_conv2dk3_silu_bias.cc:
//   sum = bias_i32[oc]; sum += wts * acts; sum = banker_srs(sum, rs);
//   sum = clamp_i8(sum); out = silu_lut[sum + 128];
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// OIYXI8O8: (O/8, I/8, kH, kW, 8 [I-inner], 8 [O-inner]) flat.
static inline int wts_idx_oiyxi8o8(int oc_full, int ic_full, int ky, int kx,
                                   int in_c, int kH, int kW) {
  int oc_t = oc_full >> 3;   // / 8
  int oc_i = oc_full & 7;    // % 8
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) + ic_i * 8 + oc_i;
}

static void
yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_scalar(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts, int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t /*padding*/) {
  event0();

  const int32_t output_width = input_width / 2;
  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);

  for (int oc_full = 0; oc_full < output_channels; oc_full++) {
    const int32_t bias_init = bias[oc_full];

    for (int x = 0; x < output_width; x++) {
      int32_t sum = bias_init;
      for (int ic_full = 0; ic_full < input_channels; ic_full++) {
        for (int ki = 0; ki < kernel_width; ki++) {
          int col = 2 * x - 1 + ki;
          if (col < 0 || col >= input_width) continue;
          int in_indx = col * input_channels + ic_full;

          int w0 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 0, ki, input_channels, kernel_height, kernel_width)];
          int w1 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 1, ki, input_channels, kernel_height, kernel_width)];
          int w2 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 2, ki, input_channels, kernel_height, kernel_width)];

          if (!skip_top) sum += line0[in_indx] * w0;
          sum += line1[in_indx] * w1;
          if (!skip_bot) sum += line2[in_indx] * w2;
        }
      }
      int32_t sum_srs = banker_srs(sum, right_shift);
      sum_srs = sum_srs > I8_MAX ? I8_MAX : (sum_srs < I8_MIN ? I8_MIN : sum_srs);
      int8_t silu_out = silu_lut[sum_srs + 128];
      output[x * output_channels + oc_full] = silu_out;
    }
  }

  event1();
}

extern "C" {

void yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_i8_i8(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts, int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t padding) {
  yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_scalar(
      line0, line1, line2, wts, bias, silu_lut, output,
      input_width, input_channels, output_channels,
      kernel_width, kernel_height, border, right_shift, padding);
}

} // extern "C"

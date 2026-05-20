//===- yolo_m0_conv2dk3_silu_bias.cc -----------------------------*- C++ -*-===//
//
// Forked from mlir-aie/aie_kernels/aie2/bottleneck/bn_conv2dk3.cc.
// See yolo_m0_conv2dk3_silu_bias.h for the list of deltas vs the upstream.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "yolo_m0_conv2dk3_silu_bias.h"

// INT8 output range. Upstream clamped to uint8 [0, 255] (acting as fused ReLU);
// we clamp to int8 [-128, 127] because SiLU is signed.
static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

// Banker rounding right-shift (matches the sim's iron_sim_kernels banker_shift).
static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  // (sum + (1 << (rs-1)) - 1 + ((sum >> rs) & 1)) >> rs
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// SiLU is applied via a 256-entry int8 LUT passed in at call time. The LUT is
// pre-computed by gen_yolo_silu_luts.py from the ONNX SiLU-chain scales
// (conv_out / scaled_hsig / silu_out) so the table encodes the full
// HardSigmoid*Mul*Mul math used by the Quark XINT8 model.

// Weight layout: raw OIYX (out_c, in_c, kH, kW), int8. m0's in_c=3 is not
// 8-aligned so weights are NOT pre-packed to OIYXI8O8 (m1/m3/m5/m7 will be,
// but those need a different kernel variant). Index per
// iron_sim_kernels._m0_conv3x3_stride2.
//
// Activation layout: (in_w, 1, in_c) int8 flat row → stride for column is
// in_c, stride for channel is 1. m0 always has in_c == 8.
//
// Border semantics (matches iron_sim_kernels._conv3x3_stride2_math):
//   border == 0 : treat top row as all-zero (line0 contributions skipped)
//   border == 1 : all 3 rows valid
//   border == 2 : treat bottom row as all-zero (line2 contributions skipped)
// The m0 builder only uses 0 (preamble) and 1 (middle); 2 is supported for
// general stride-2 blocks but unused here.

static inline int wts_idx_oiyx(int oc_full, int ic_full, int ky, int kx,
                               int in_c, int kH, int kW) {
  return ((oc_full * in_c + ic_full) * kH + ky) * kW + kx;
}

//*****************************************************************************
// 3x3 stride-2 conv, scalar.
// act: int8, wts: int8 (OIYX raw), bias: int32 (accum init), out: int8.
//*****************************************************************************
static void
yolo_m0_conv2dk3_i8_stride2_silu_bias_scalar(
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
      // 3x3 taps centered on input column (2*x): ki in {0,1,2} → col 2*x-1, 2*x, 2*x+1.
      // Out-of-range cols are zero-padded (matches sim's `valid` mask).
      int32_t sum = bias_init;
      for (int ic_full = 0; ic_full < input_channels; ic_full++) {
        for (int ki = 0; ki < kernel_width; ki++) {
          int col = 2 * x - 1 + ki;
          if (col < 0 || col >= input_width) continue;
          int in_indx = col * input_channels + ic_full;

          int w0 = wts[wts_idx_oiyx(oc_full, ic_full, 0, ki, input_channels, kernel_height, kernel_width)];
          int w1 = wts[wts_idx_oiyx(oc_full, ic_full, 1, ki, input_channels, kernel_height, kernel_width)];
          int w2 = wts[wts_idx_oiyx(oc_full, ic_full, 2, ki, input_channels, kernel_height, kernel_width)];

          if (!skip_top) sum += line0[in_indx] * w0;
          sum += line1[in_indx] * w1;
          if (!skip_bot) sum += line2[in_indx] * w2;
        }
      }
      int32_t sum_srs = banker_srs(sum, right_shift);
      sum_srs = sum_srs > I8_MAX ? I8_MAX : (sum_srs < I8_MIN ? I8_MIN : sum_srs);
      // SiLU LUT lookup, indexed by pre_silu + 128 (int8 → 0..255).
      int8_t silu_out = silu_lut[sum_srs + 128];
      // Output layout: (out_w, 1, out_c) flat → stride for column is out_c.
      output[x * output_channels + oc_full] = silu_out;
    }
  }

  event1();
}

extern "C" {

void yolo_m0_conv2dk3_stride2_silu_bias_i8_i8(
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
  yolo_m0_conv2dk3_i8_stride2_silu_bias_scalar(
      line0, line1, line2, wts, bias, silu_lut, output,
      input_width, input_channels, output_channels,
      kernel_width, kernel_height, border, right_shift, padding);
}

} // extern "C"

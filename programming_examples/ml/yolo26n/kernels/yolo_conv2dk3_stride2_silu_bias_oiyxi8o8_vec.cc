//===- yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc -----------------*- C++ -*-===//
//
// Vectorized 3x3 stride-2 conv with OIYXI8O8 weight layout, drop-in
// replacement for yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.cc on AIE2P.
//
// Same API + same bit-exact numerics as the scalar version:
//   acc[oc] = bias[oc]
//   for ic, ky, kx: acc[oc] += line[ky][col*in_c+ic] * wts[oiyxi8o8(...)]
//   acc -> banker_srs(rs) -> clamp_i8 -> silu_lut[clamped + 128]
//
// Inner reduction uses aie::mmul<4, 8, 8, int8, int8> for 4 stride-2
// output pixels x 8 oc_inner per mmul call. Input gather is per-pixel
// scalar loads + insert<8> into the 32-byte A operand — preserves the
// existing HWC activation layout (no IRON design changes needed).
//
// A future Phase 1b can move to ic-tile-major activations + aie::shuffle
// based gather for an additional 1.5-2x on top of this kernel's win.
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

// OIYXI8O8 weight base offset for one (oc_tile, ic_tile, ky, kx). The 64
// bytes at this offset are arranged [I_inner=8][O_inner=8] in row-major,
// which matches aie::mmul<*, 8, 8> B operand expectation exactly.
static inline int wts_tile_off(int oc_tile, int ic_tile, int ky, int kx,
                               int ic_tiles, int kH, int kW) {
  return (((oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

// Reference scalar weight index (for tail / fallback paths).
static inline int wts_idx_oiyxi8o8(int oc_full, int ic_full, int ky, int kx,
                                   int in_c, int kH, int kW) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) + ic_i * 8 + oc_i;
}

static void yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_vec(
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

  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = input_channels / 8;
  const int oc_tiles = output_channels / 8;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  int8_t *line[3] = {line0, line1, line2};

  // Process output_width in tiles of 4 stride-2 pixels = 8 input cols.
  const int x_tiles = output_width / 4;

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      // Zero-init accumulator. Bias is folded into the scalar SRS epilogue
      // below (since SRS+clamp+LUT lookup is already scalar per output).
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      // Output pixel base column for this tile.
      const int x_out_base = x_tile * 4;
      // Leftmost input column index (output col 0's kx=0 reaches here).
      const int x_in_base = 2 * x_out_base - 1;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];

          for (int kx = 0; kx < kernel_width; ++kx) {
            // Build the 32-byte A operand: 4 stride-2 pixels x 8 ic_inner.
            // Pixel p (p in 0..3) at column col = x_in_base + 2*p + kx.
            // If col is out of bounds, zero-fill that 8-byte slot (zero border).
            // Scalar byte assemblage; compiler should vectorize the 8-byte
            // memcpys into vector loads in the contiguous case.
            int8_t a_buf[32];
            bool any_valid = false;
            for (int p = 0; p < 4; ++p) {
              int col = x_in_base + 2 * p + kx;
              if (col < 0 || col >= input_width) {
                for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = 0;
              } else {
                int8_t *src = line_ptr + col * input_channels + ic_t * 8;
                for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
                any_valid = true;
              }
            }
            if (!any_valid) continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

            // 64-byte B operand: OIYXI8O8 for this (oc_t, ic_t, ky, kx).
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, ic_tiles,
                                       kernel_height, kernel_width);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);

            acc.mac(in_a, in_b);
          }
        }
      }

      // Drain accumulator to 32 int32s, fold bias in, scalar SRS + clamp + LUT.
      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias[oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          output[x_out * output_channels + oc_t * 8 + j] = silu_lut[sr + 128];
        }
      }
    }

    // Tail outputs if output_width is not a multiple of 4: scalar fallback.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < input_channels; ++ic_full) {
          for (int ki = 0; ki < kernel_width; ++ki) {
            int col = 2 * x - 1 + ki;
            if (col < 0 || col >= input_width) continue;
            int in_indx = col * input_channels + ic_full;
            int w0 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 0, ki,
                                          input_channels, kernel_height, kernel_width)];
            int w1 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 1, ki,
                                          input_channels, kernel_height, kernel_width)];
            int w2 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 2, ki,
                                          input_channels, kernel_height, kernel_width)];
            if (!skip_top) sum += line0[in_indx] * w0;
            sum += line1[in_indx] * w1;
            if (!skip_bot) sum += line2[in_indx] * w2;
          }
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        output[x * output_channels + oc_full] = silu_lut[sr + 128];
      }
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
  yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_vec(
      line0, line1, line2, wts, bias, silu_lut, output,
      input_width, input_channels, output_channels,
      kernel_width, kernel_height, border, right_shift, padding);
}

} // extern "C"

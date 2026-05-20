//===- yolo_m0_conv2dk3_silu_bias_vec.cc ---------------------------*- C++ -*-===//
//
// Vectorized stem kernel for m0: 3x3 stride-2 INT8 conv, in_c=3 (padded to 8),
// raw OIYX weight layout (NOT OIYXI8O8 because in_c=3 is not 8-aligned).
// Drop-in .o-level replacement for yolo_m0_conv2dk3_silu_bias.cc.
//
// Inner reduction uses aie::mmul<4, 8, 8, int8, int8>. The B operand is
// built once per (oc_tile, ky, kx) from the OIYX raw weights, packing
// 8 ic_inner (only 3 valid + 5 zero-padded) x 8 oc_inner into a 64-byte
// vector. Padding for in_c is host-managed (the IRON design hands us
// (in_w, 1, 8) int8 inputs with the high 5 channels zero), so the kernel
// can treat in_c=8 throughout.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "yolo_m0_conv2dk3_silu_bias.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// OIYX raw weight index: byte[oc][ic][ky][kx].
static inline int wts_idx_oiyx(int oc_full, int ic_full, int ky, int kx,
                               int in_c, int kH, int kW) {
  return ((oc_full * in_c + ic_full) * kH + ky) * kW + kx;
}

static void
yolo_m0_conv2dk3_i8_stride2_silu_bias_vec(
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

  // m0: input_channels is the *padded* width (= 8). The OIYX raw weights
  // index uses the ORIGINAL channel count (= 3), but the IRON design
  // pre-zeros channels 3..7 in the input so we can compute with the
  // padded width and rely on those zeros to make ic 3..7 contributions
  // vanish. To keep things simple here, we pack the weight into 8 ic_inner
  // slots (loading 3 real + 5 zero-padded) and treat the conv as in_c=8.
  const int oc_tiles = output_channels / 8;
  const int x_tiles = output_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  int8_t *line[3] = {line0, line1, line2};

  // For each oc_tile and each (ky, kx), pre-pack the 8 ic_inner x 8 oc_inner
  // weight tile as a 64-byte vector. Weights have shape (out_c, 3, 3, 3) with
  // in_c=3 — we pack the first 3 ic_inner slots from OIYX bytes and zero the
  // remaining 5. Pack lazily inside the loop to avoid a large prepass buffer.

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    // Pre-pack the 9 (ky, kx) weight tiles for this oc_tile into a contiguous
    // 9 * 64-byte buffer. Reused across all x_tiles.
    // OIYX is sized (out_c, in_c_padded=input_channels=8, kH, kW). The host
    // pre-pads channels 3..7 with zeros — we read the full 8-channel buffer.
    alignas(32) int8_t wbuf[9 * 64];
    for (int ky = 0; ky < kernel_height; ++ky) {
      for (int kx = 0; kx < kernel_width; ++kx) {
        int wt_off = (ky * kernel_width + kx) * 64;
        for (int ii = 0; ii < 8; ++ii) {
          for (int oo = 0; oo < 8; ++oo) {
            int oc_full = oc_t * 8 + oo;
            wbuf[wt_off + ii * 8 + oo] = wts[wts_idx_oiyx(
                oc_full, ii, ky, kx, input_channels, kernel_height, kernel_width)];
          }
        }
      }
    }

    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      const int x_out_base = x_tile * 4;
      const int x_in_base = 2 * x_out_base - 1;

      for (int ky = ky_start; ky < ky_end; ++ky) {
        int8_t *line_ptr = line[ky];

        for (int kx = 0; kx < kernel_width; ++kx) {
          alignas(32) int8_t a_buf[32];
          bool any_valid = false;
          for (int p = 0; p < 4; ++p) {
            int col = x_in_base + 2 * p + kx;
            if (col < 0 || col >= input_width) {
              for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = 0;
            } else {
              int8_t *src = line_ptr + col * input_channels;
              for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
              any_valid = true;
            }
          }
          if (!any_valid) continue;
          aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

          int wt_off = (ky * kernel_width + kx) * 64;
          aie::vector<int8, 64> in_b = aie::load_v<64>(&wbuf[wt_off]);
          acc.mac(in_a, in_b);
        }
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          int oc_full = oc_t * 8 + j;
          int32_t s = acc_vec[p * 8 + j] + bias[oc_full];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          output[x_out * output_channels + oc_full] = silu_lut[sr + 128];
        }
      }
    }

    // Tail outputs scalar fallback (in case output_width % 4 != 0).
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < input_channels; ++ic_full) {
          for (int kx = 0; kx < kernel_width; ++kx) {
            int col = 2 * x - 1 + kx;
            if (col < 0 || col >= input_width) continue;
            int in_indx = col * input_channels + ic_full;
            int w0 = wts[wts_idx_oiyx(oc_full, ic_full, 0, kx, input_channels,
                                      kernel_height, kernel_width)];
            int w1 = wts[wts_idx_oiyx(oc_full, ic_full, 1, kx, input_channels,
                                      kernel_height, kernel_width)];
            int w2 = wts[wts_idx_oiyx(oc_full, ic_full, 2, kx, input_channels,
                                      kernel_height, kernel_width)];
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
  yolo_m0_conv2dk3_i8_stride2_silu_bias_vec(
      line0, line1, line2, wts, bias, silu_lut, output,
      input_width, input_channels, output_channels,
      kernel_width, kernel_height, border, right_shift, padding);
}

} // extern "C"

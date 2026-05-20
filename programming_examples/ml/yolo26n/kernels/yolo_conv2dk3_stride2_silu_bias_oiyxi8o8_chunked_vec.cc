//===- yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc -----*- C++ -*-===//
//
// Vectorized chunked variant of the OIYXI8O8 stride-2 conv. Same math as
// the non-chunked vec kernel (yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc),
// but the weight buffer holds only one chunk of output channels and the
// kernel writes to the global output row at oc_offset + chunk_oc.
//
// One .cc file produces three .o (m3, m5, m7) via -DKERNEL_SUFFIX=_mN.
// Drop-in .o-level replacement for the scalar chunked .o.
//
// Inner reduction uses aie::mmul<4, 8, 8, int8, int8>; bias folded into
// the scalar SRS+LUT epilogue.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// Per-block symbol mangling. Compile per-block with -DKERNEL_SUFFIX=_mN.
#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// Chunk-local OIYXI8O8 weight base offset for (chunk_oc_tile, ic_tile, ky, kx).
// chunk_oc_tile = (chunk-local oc_full) / 8, in [0..oc_count/8).
static inline int wts_chunk_tile_off(int chunk_oc_tile, int ic_tile, int ky, int kx,
                                     int ic_tiles, int kH, int kW) {
  return (((chunk_oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

// Reference scalar weight index (tail path).
static inline int wts_chunk_idx_oiyxi8o8(int chunk_oc_full, int ic_full,
                                          int ky, int kx,
                                          int in_c, int kH, int kW) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) + ic_i * 8 + oc_i;
}

static void
yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_chunked_vec(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts_chunk, int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t oc_offset,
    const int32_t oc_count) {
  event0();

  const int32_t output_width = input_width / 2;
  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);

  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = input_channels / 8;
  const int chunk_oc_tiles = oc_count / 8;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  int8_t *line[3] = {line0, line1, line2};

  const int x_tiles = output_width / 4;

  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    // Global oc range for this chunk_oc_t: oc_offset + chunk_oc_t*8 .. +7.
    const int oc_full_base = oc_offset + chunk_oc_t * 8;

    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      const int x_out_base = x_tile * 4;
      const int x_in_base = 2 * x_out_base - 1;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
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
                int8_t *src = line_ptr + col * input_channels + ic_t * 8;
                for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
                any_valid = true;
              }
            }
            if (!any_valid) continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

            int wts_off = wts_chunk_tile_off(chunk_oc_t, ic_t, ky, kx,
                                             ic_tiles, kernel_height, kernel_width);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);

            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          int oc_full = oc_full_base + j;
          int32_t s = acc_vec[p * 8 + j] + bias[oc_full];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          // Write to FULL-row offset (output_channels stride, global oc_full).
          output[x_out * output_channels + oc_full] = silu_lut[sr + 128];
        }
      }
    }

    // Tail outputs if output_width is not a multiple of 4: scalar fallback.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int chunk_oc_full = chunk_oc_t * 8 + j;
        int oc_full = oc_offset + chunk_oc_full;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < input_channels; ++ic_full) {
          for (int ki = 0; ki < kernel_width; ++ki) {
            int col = 2 * x - 1 + ki;
            if (col < 0 || col >= input_width) continue;
            int in_indx = col * input_channels + ic_full;
            int w0 = wts_chunk[wts_chunk_idx_oiyxi8o8(chunk_oc_full, ic_full, 0, ki,
                                                       input_channels, kernel_height, kernel_width)];
            int w1 = wts_chunk[wts_chunk_idx_oiyxi8o8(chunk_oc_full, ic_full, 1, ki,
                                                       input_channels, kernel_height, kernel_width)];
            int w2 = wts_chunk[wts_chunk_idx_oiyxi8o8(chunk_oc_full, ic_full, 2, ki,
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

void KERNEL_NAME(yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts_chunk, int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t oc_offset,
    const int32_t oc_count) {
  yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_chunked_vec(
      line0, line1, line2, wts_chunk, bias, silu_lut, output,
      input_width, input_channels, output_channels,
      kernel_width, kernel_height, border, right_shift,
      oc_offset, oc_count);
}

} // extern "C"

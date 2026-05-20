//===- yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked.cc --------*- C++ -*-===//
//
// Scalar chunked variant of the OIYXI8O8 stride-2 conv. See the .h for
// chunking semantics + index math.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// Per-block symbol mangling (see yolo_c3k2_small_cv1_split.cc for context).
// Compile per-block with -DKERNEL_SUFFIX=_mN so m3/m5/m7 each get a unique
// exported symbol; lets them coexist in a chained xclbin.
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

// Chunk is (oc_count/8, in_c/8, kH, kW, 8, 8) OIYXI8O8 packed.
// Use chunk-local oc index = oc_full - oc_offset.
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
yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_chunked_scalar(
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

  for (int chunk_oc = 0; chunk_oc < oc_count; chunk_oc++) {
    const int oc_full = oc_offset + chunk_oc;
    const int32_t bias_init = bias[oc_full];

    for (int x = 0; x < output_width; x++) {
      int32_t sum = bias_init;
      for (int ic_full = 0; ic_full < input_channels; ic_full++) {
        for (int ki = 0; ki < kernel_width; ki++) {
          int col = 2 * x - 1 + ki;
          if (col < 0 || col >= input_width) continue;
          int in_indx = col * input_channels + ic_full;

          int w0 = wts_chunk[wts_chunk_idx_oiyxi8o8(chunk_oc, ic_full, 0, ki, input_channels, kernel_height, kernel_width)];
          int w1 = wts_chunk[wts_chunk_idx_oiyxi8o8(chunk_oc, ic_full, 1, ki, input_channels, kernel_height, kernel_width)];
          int w2 = wts_chunk[wts_chunk_idx_oiyxi8o8(chunk_oc, ic_full, 2, ki, input_channels, kernel_height, kernel_width)];

          if (!skip_top) sum += line0[in_indx] * w0;
          sum += line1[in_indx] * w1;
          if (!skip_bot) sum += line2[in_indx] * w2;
        }
      }
      int32_t sum_srs = banker_srs(sum, right_shift);
      sum_srs = sum_srs > I8_MAX ? I8_MAX : (sum_srs < I8_MIN ? I8_MIN : sum_srs);
      int8_t silu_out = silu_lut[sum_srs + 128];
      // Write to FULL-row offset (output_channels stride, oc_full position).
      output[x * output_channels + oc_full] = silu_out;
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
  yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_chunked_scalar(
      line0, line1, line2, wts_chunk, bias, silu_lut, output,
      input_width, input_channels, output_channels,
      kernel_width, kernel_height, border, right_shift,
      oc_offset, oc_count);
}

} // extern "C"

//===- yolo_c3k2_heavy_inner_pair_cv1_streamed.cc -------------*- C++ -*-===//
//
// Chunked-OC variant of yolo_c3k2_heavy_inner_pair_cv1. 3x3 stride-1
// conv + SiLU LUT, called n_chunks times per output row to process
// chunk_oc output channels at a time. Bias is the full out_c array
// (slice in-kernel). Output buffer is the full row; this call writes
// the chunk_idx slice.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

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

// Chunk OIYXI8O8 indexer.
static inline int wts_chunk_idx(int chunk_oc, int ic_full, int ky, int kx,
                                 int in_c, int kH, int kW) {
  int oc_t = chunk_oc >> 3;
  int oc_i = chunk_oc & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) +
         ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t n_chunks,
    const int32_t chunk_idx) {
  event0();

  const int32_t chunk_oc = output_channels / n_chunks;
  const int32_t oc_offset = chunk_idx * chunk_oc;
  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);

  for (int co = 0; co < chunk_oc; co++) {
    const int32_t binit = bias_full[oc_offset + co];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        for (int kx = 0; kx < kernel_width; kx++) {
          int col = x - 1 + kx;
          if (col < 0 || col >= input_width) continue;
          int in_indx = col * input_channels + ic;

          int w0 = wts_chunk[wts_chunk_idx(co, ic, 0, kx, input_channels,
                                           kernel_height, kernel_width)];
          int w1 = wts_chunk[wts_chunk_idx(co, ic, 1, kx, input_channels,
                                           kernel_height, kernel_width)];
          int w2 = wts_chunk[wts_chunk_idx(co, ic, 2, kx, input_channels,
                                           kernel_height, kernel_width)];

          if (!skip_top) sum += line0[in_indx] * w0;
          sum += line1[in_indx] * w1;
          if (!skip_bot) sum += line2[in_indx] * w2;
        }
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      output[x * output_channels + (oc_offset + co)] = silu_lut[s + 128];
    }
  }

  event1();
}

} // extern "C"

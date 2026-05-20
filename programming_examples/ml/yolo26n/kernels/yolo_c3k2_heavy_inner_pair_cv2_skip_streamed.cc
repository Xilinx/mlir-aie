//===- yolo_c3k2_heavy_inner_pair_cv2_skip_streamed.cc --------*- C++ -*-===//
//
// Chunked-OC variant of yolo_c3k2_heavy_inner_pair_cv2_skip. 3x3
// stride-1 conv + SiLU LUT + CROSS-SCALE skip-add. Called n_chunks
// times per output row to process chunk_oc output channels. Skip-add
// formula matches the non-streamed kernel:
//     out_i8 = banker_srs(y * y_mult + cv2silu * cv2_mult, rsh_add)
// All four m6/m8 pair adds use y_mult=1, cv2_mult=2, rsh_add=1.
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

void KERNEL_NAME(yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int8_t *skip_row,
    int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t n_chunks,
    const int32_t chunk_idx,
    const int32_t skip_y_mult,
    const int32_t skip_cv2_mult,
    const int32_t skip_rsh_add) {
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
      int32_t cv2silu = silu_lut[s + 128];
      // skip_row has chunk_oc channels relative to oc_offset? No — skip_row
      // is the FULL pair input row (output_channels wide). Index by absolute oc.
      int32_t y = (int32_t)skip_row[x * output_channels + (oc_offset + co)];
      int32_t sum_pre = y * skip_y_mult + cv2silu * skip_cv2_mult;
      int32_t added = banker_srs(sum_pre, skip_rsh_add);
      added = added > I8_MAX ? I8_MAX : (added < I8_MIN ? I8_MIN : added);
      output[x * output_channels + (oc_offset + co)] = (int8_t)added;
    }
  }

  event1();
}

} // extern "C"

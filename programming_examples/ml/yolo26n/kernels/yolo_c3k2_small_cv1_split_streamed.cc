//===- yolo_c3k2_small_cv1_split_streamed.cc -------------------*- C++ -*-===//
//
// Chunked-OC variant of yolo_c3k2_small_cv1_split. Called n_chunks times
// per row; each call processes chunk_oc = twoc/n_chunks output channels
// from a single chunk of OIYXI8O8-packed weights. Bias is the FULL twoc
// array (we slice the chunk's portion in-kernel). SiLU LUT epilogue
// before writing to out_top or out_bot depending on whether this chunk
// falls in the first or second half of the output channel axis.
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

// Chunk is OIYXI8O8 packed for (chunk_oc, in_c, 1, 1).
static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv1_split_streamed_silu_bias_i8_i8)(
    int8_t *in_row,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int8_t *out_top,
    int8_t *out_bot_a,    // bot copy for m_0_split consumer
    int8_t *out_bot_b,    // bot copy for cv2 consumer (eliminates bot fanout)
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t twoc,
    const int32_t n_chunks,
    const int32_t chunk_idx,
    const int32_t right_shift) {
  event0();

  const int32_t chunk_oc = twoc / n_chunks;
  const int32_t c = twoc >> 1;
  const int32_t chunks_per_half = n_chunks >> 1;
  const bool is_top = (chunk_idx < chunks_per_half);
  const int32_t dst_oc_offset = is_top
      ? chunk_idx * chunk_oc
      : (chunk_idx - chunks_per_half) * chunk_oc;
  const int32_t bias_offset = chunk_idx * chunk_oc;

  for (int co = 0; co < chunk_oc; co++) {
    const int32_t binit = bias_full[bias_offset + co];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts_chunk[wts_chunk_idx_1x1(co, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      int8_t silu = silu_lut[s + 128];
      const int idx = x * c + (dst_oc_offset + co);
      if (is_top) {
        out_top[idx] = silu;
      } else {
        out_bot_a[idx] = silu;
        out_bot_b[idx] = silu;
      }
    }
  }

  event1();
}

} // extern "C"

//===- yolo_m9_cv2_concat2_streamed.cc -------------------------*- C++ -*-===//
//
// Streamed (chunked-OC) cv2 1x1 conv for the final m9 mixing layer.
// Input is the channel-concat of (a, ffn_block_out):
//     a              : (in_w, c=128)  — cv1 top half (skip), prefetched
//                                       into a (in_h, in_w, c) cache
//     ffn_block_out  : (in_w, c=128)  — stage 9 output, streamed per row
// Logical conv input: (in_w, 2*c=256) where chans [0, c) = a and
// chans [c, 2*c) = ffn_block_out. Output (in_w, out_c=256) at scale 2^-?
// with SiLU+bias applied. Weights streamed in N chunks of chunk_oc out
// channels (OIYXI8O8 packed).
//
// Symbol is m9-specific.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m9_cv2_concat2_streamed_silu_bias_i8_i8(
    int8_t *top_cache,         // (in_h, in_w, c) prefetched cv1 top half
    int8_t *ffn_row,           // (in_w, c)
    int8_t *wts_chunk,         // OIYXI8O8 chunk (chunk_oc, 2*c, 1, 1)
    int32_t *bias_full,        // (out_c=2*c,)
    int8_t *silu_lut,          // (256,)
    int8_t *out_row,           // (in_w, out_c)
    const int32_t yi,
    const int32_t input_width,
    const int32_t c_per_half,         // 128
    const int32_t out_c,              // 256
    const int32_t n_chunks,
    const int32_t chunk_idx,
    const int32_t right_shift) {
  event0();

  const int32_t in_c = 2 * c_per_half;             // 256
  const int32_t chunk_oc = out_c / n_chunks;
  const int32_t dst_oc_offset = chunk_idx * chunk_oc;
  const int32_t bias_offset = chunk_idx * chunk_oc;

  int8_t *top_row = top_cache + yi * input_width * c_per_half;

  for (int co = 0; co < chunk_oc; co++) {
    const int32_t binit = bias_full[bias_offset + co];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      // First c_per_half channels come from top_cache.
      for (int ic = 0; ic < c_per_half; ic++) {
        sum += top_row[x * c_per_half + ic] *
               wts_chunk[wts_chunk_idx_1x1(co, ic, in_c)];
      }
      // Next c_per_half channels come from ffn_row.
      for (int ic = 0; ic < c_per_half; ic++) {
        sum += ffn_row[x * c_per_half + ic] *
               wts_chunk[wts_chunk_idx_1x1(co, c_per_half + ic, in_c)];
      }
      int32_t s = banker_srs(sum, right_shift);
      if (s > I8_MAX) s = I8_MAX;
      if (s < I8_MIN) s = I8_MIN;
      out_row[x * out_c + (dst_oc_offset + co)] = silu_lut[s + 128];
    }
  }

  event1();
}

} // extern "C"

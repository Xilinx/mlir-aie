//===- yolo_m9_ffn_1_skip_row.cc ------------------------------*- C++ -*-===//
//
// 1x1 i8 conv 256 -> 128 + plain (same-scale) skip-add with attn_block_out.
// Both ffn.1 output and the skip are at scale 2^-4 per ONNX, so the add
// is just int + clip (no cross-scale shift needed).
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

static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m9_ffn_1_skip_row_i8_i8(
    int8_t *mid_row,           // (in_w, in_c=256)
    int8_t *wts,               // (out_c, in_c, 1, 1) OIYXI8O8
    int32_t *bias,             // (out_c,)
    int8_t *skip_row,          // (in_w, 1, out_c) attn_block_out skip (same shape as bot_fifo)
    int8_t *out_row,           // (in_w, out_c)
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t right_shift) {
  event0();

  for (int oc = 0; oc < output_channels; oc++) {
    const int32_t binit = bias[oc];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += mid_row[x * input_channels + ic] *
               wts[wts_idx_oiyxi8o8_1x1(oc, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift);
      if (s > I8_MAX) s = I8_MAX;
      if (s < I8_MIN) s = I8_MIN;
      // Same-scale skip-add: just sum and clip.
      int32_t add = s + (int32_t)skip_row[x * output_channels + oc];
      if (add > I8_MAX) add = I8_MAX;
      if (add < I8_MIN) add = I8_MIN;
      out_row[x * output_channels + oc] = (int8_t)add;
    }
  }

  event1();
}

} // extern "C"

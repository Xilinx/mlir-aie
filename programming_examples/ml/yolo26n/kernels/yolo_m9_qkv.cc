//===- yolo_m9_qkv.cc ----------------------------------------*- C++ -*-===//
//
// Scalar 1x1 INT8 conv 128 -> 256 (no SiLU; activation=None per yolo_spec).
// Bias initializes the accumulator. Output is a single (in_w, 256) row.
//
// Note on the PSA packed-frame layout: the 256 output channels are split
// 128+128 per head (head 0 = chans [0,128), head 1 = chans [128,256)),
// and within each head the first 32 = Q, next 32 = K, last 64 = V. That
// packing is a pure layout reinterpretation of the natural (in_h, in_w, 256)
// conv output — attn_core's pre-pass can do the index reshuffle in tile L1.
// For stage 2 we emit the natural per-row layout so the oracle is a plain
// 1x1 conv (no packing). Stage 3 introduces the per-head pack.
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

void yolo_m9_qkv_i8_i8(
    int8_t *in_row,
    int8_t *wts,
    int32_t *bias,
    int8_t *out_row,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  for (int oc = 0; oc < output_channels; oc++) {
    const int32_t binit = bias[oc];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts[wts_idx_oiyxi8o8_1x1(oc, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      out_row[x * output_channels + oc] = (int8_t)s;
    }
  }

  event1();
}

} // extern "C"

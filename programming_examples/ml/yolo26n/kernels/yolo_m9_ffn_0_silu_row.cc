//===- yolo_m9_ffn_0_silu_row.cc ------------------------------*- C++ -*-===//
//
// Streamed (chunked-OC) 1x1 i8 conv 128 -> 256 + SiLU LUT + bias for the
// PSA ffn first conv. ffn.0's full weights (256*128 = 32KB) don't fit on
// the ffn_tile alongside ffn.1's 32KB and a mid scratch + I/O fifos, so
// we stream the weights in chunks (4 × 8KB = chunk_oc=64 output chans
// per chunk). Output is the full mid buffer (in_w=16, out_c=256) written
// piecewise per chunk.
//
// Output is i8 at scale 2^-4 (ffn.0/act/Mul QL scale per ONNX).
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

void yolo_m9_ffn_0_silu_row_i8_i8(
    int8_t *in_row,            // (in_w, in_c=128)
    int8_t *wts_chunk,         // OIYXI8O8 packed chunk (chunk_oc, in_c, 1, 1)
    int32_t *bias_full,        // (out_c,)
    int8_t *silu_lut,          // (256,)
    int8_t *mid_out,           // (in_w, out_c) — chunk writes its OC slice
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t out_c,            // 256
    const int32_t n_chunks,         // 4
    const int32_t chunk_idx,
    const int32_t right_shift) {
  event0();

  const int32_t chunk_oc = out_c / n_chunks;       // 64
  const int32_t dst_oc_offset = chunk_idx * chunk_oc;
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
      if (s > I8_MAX) s = I8_MAX;
      if (s < I8_MIN) s = I8_MIN;
      mid_out[x * out_c + (dst_oc_offset + co)] = silu_lut[s + 128];
    }
  }

  event1();
}

} // extern "C"

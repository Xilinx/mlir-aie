//===- yolo_m9_qkv_pack.cc -----------------------------------*- C++ -*-===//
//
// Per-head packer for the PSA attn input. Consumes one natural-layout
// (in_w, twoc=256) row of qkv output and writes its contribution into the
// packed per-head frame: (kd+kd+hd=128, N=256), where N = row_idx*in_w + x.
//
// Two heads share the qkv output's 256 channels: head 0 = chans [0,128),
// head 1 = chans [128,256). This kernel packs ONE head per call — the
// caller selects head via head_idx (0 or 1) and routes to the correct
// frame buffer. The 128 channels within each head are already in
// (Q[0..31] | K[32..63] | V[64..127]) order from the qkv conv, so the
// packing here is just an index transpose: input[x, head*128 + s] ->
// frame[s, row_idx*in_w + x].
//
// Symbol is m9-specific; m9 is the sole consumer.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

extern "C" {

void yolo_m9_qkv_pack_i8_i8(
    int8_t *in_row,           // (in_w, twoc) natural-layout qkv row
    int8_t *head_frame,       // (head_slots, N) destination per-head slice
    const int32_t input_width,        // in_w (=16)
    const int32_t twoc,                // total qkv chans (=256)
    const int32_t head_slots,          // chans to COPY per call (e.g. 128 full, 64 Q+K only)
    const int32_t head_stride,         // chans per full head in input (=128 for m9 qkv)
    const int32_t N,                   // tokens per sample (or in_w for per-row chunk)
    const int32_t head_idx,            // 0 or 1
    const int32_t row_idx) {           // 0..in_h-1 (or 0 for per-row chunk mode)
  event0();

  const int32_t chan_offset = head_idx * head_stride;
  const int32_t n_base = row_idx * input_width;

  for (int s = 0; s < head_slots; s++) {
    for (int x = 0; x < input_width; x++) {
      head_frame[s * N + (n_base + x)] =
          in_row[x * twoc + (chan_offset + s)];
    }
  }

  event1();
}

} // extern "C"

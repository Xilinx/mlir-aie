//===- yolo_m9_qk_pack.cc ------------------------------------*- C++ -*-===//
//
// Per-head Q+K extractor. Identical math to yolo_m9_qkv_pack but with
// (qk_slots=64, head_stride=128) defaults baked in conceptually — the
// kernel signature is fully generic so the IRON-side caller can use a
// different per-head buffer shape than qkv_pack does. Used by stages
// that fold pack + qk matmul into a single attn_core worker.
//
// Symbol distinct from qkv_pack so each gets its own .o with the
// shape signature its IRON-side caller expects (MLIR func sym ↔ .o
// symbol; one .o = one shape signature).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

extern "C" {

void yolo_m9_qk_pack_i8_i8(
    int8_t *in_row,           // (in_w, twoc) natural-layout qkv row
    int8_t *qk_frame,         // (qk_slots, N) destination per-head buffer
    const int32_t input_width,        // in_w (=16)
    const int32_t twoc,                // total qkv chans (=256)
    const int32_t qk_slots,            // chans to COPY per call (=64 for Q+K only)
    const int32_t head_stride,         // chans per full head in input (=128)
    const int32_t N,                   // dst N dimension (=in_h*in_w=256)
    const int32_t head_idx,            // 0 or 1
    const int32_t row_idx) {           // 0..in_h-1
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t chan_offset = head_idx * head_stride;
  const int32_t n_base = row_idx * input_width;

  for (int s = 0; s < qk_slots; s++) {
    for (int x = 0; x < input_width; x++) {
      qk_frame[s * N + (n_base + x)] =
          in_row[x * twoc + (chan_offset + s)];
    }
  }

  event1();
}

} // extern "C"

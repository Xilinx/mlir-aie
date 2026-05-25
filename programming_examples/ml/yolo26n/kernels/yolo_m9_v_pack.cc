//===- yolo_m9_v_pack.cc -------------------------------------*- C++ -*-===//
//
// Per-head V extractor for the PSA attention pipe (sv matmul). Reads one
// natural-layout (in_w, twoc=256) qkv row and copies the V slice for a
// given head into the destination (head_dim, N) per-head V buffer at
// the appropriate row offset.
//
// V chans within each head occupy slots 64..127 of the 128-slot
// (Q || K || V) layout, so v_offset = head_idx * head_stride + 2*kd.
//
// Distinct symbol from qk_pack so each .o has its own kernel signature
// (the per-head V buffer is (64, 256) vs qk_pack's (64, 256) — same
// shape today, but keeping them separate keeps the role explicit and
// makes future fusion / vectorization easier).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

extern "C" {

void yolo_m9_v_pack_i8_i8(
    int8_t *in_row,  // (in_w, twoc) natural-layout qkv row
    int8_t *v_frame, // (head_dim, N) destination per-head V buffer
    const int32_t input_width, const int32_t twoc,
    const int32_t head_dim,         // 64
    const int32_t v_offset_in_head, // 64 (= 2*kd; V starts after Q and K)
    const int32_t head_stride,      // 128
    const int32_t N,                // 256
    const int32_t head_idx, const int32_t row_idx) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  // Hardcoded for m9 call site (in_w=16, twoc=256, head_dim=64,
  // head_stride=128, v_offset=64, N=256). Constexpr lowers addressing
  // to immediates; AIE_LOOP_RANGE hints peano on exact trip counts.
  (void)input_width;
  (void)twoc;
  (void)head_dim;
  (void)v_offset_in_head;
  (void)head_stride;
  (void)N;
  constexpr int kInW = 16;
  constexpr int kTwoC = 256;
  constexpr int kHeadDim = 64;
  constexpr int kVOff = 64;
  constexpr int kHeadStride = 128;
  constexpr int kN = 256;

  const int32_t chan_base = head_idx * kHeadStride + kVOff;
  const int32_t n_base = row_idx * kInW;

  // Transpose copy: read 64 strided i8 (stride twoc=256) per x position,
  // write to v_frame's per-channel row. Strided reads block clean vec
  // ops; scalar copy with peano's auto-pipelining is the practical
  // tightest form.
  AIE_LOOP_RANGE(kHeadDim, kHeadDim)
  for (int s = 0; s < kHeadDim; s++) {
    AIE_LOOP_RANGE(kInW, kInW)
    for (int x = 0; x < kInW; x++) {
      v_frame[s * kN + (n_base + x)] = in_row[x * kTwoC + (chan_base + s)];
    }
  }

  event1();
}

} // extern "C"

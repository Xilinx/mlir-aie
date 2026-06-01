//===- yolo_m9_pack.cc ----------------------------------------*- C++ -*-===//
//
// Unified per-head packer for the PSA attention pipe. Reads one natural-
// layout (in_w, twoc=256) qkv-conv output row and copies a configurable
// slice of channels into a destination per-head (slots, N) buffer in
// transposed layout: dst[s, row_idx*in_w + x] = src[x, chan_offset + s].
//
// Replaces 3 near-identical .cc files (qkv_pack, qk_pack, v_pack). Each
// concrete .o is produced by compiling this source with:
//   -DYOLO_M9_PACK_SYMBOL=<extern_c_symbol_name>
//   -DYOLO_M9_PACK_SLOTS=<N>                  (constexpr inner iters)
//   -DYOLO_M9_PACK_EXTRA_OFFSET=<bytes>       (extra chan offset within head)
//
// Three concrete instantiations used by m9:
//   yolo_m9_qkv_pack_i8_i8 : SLOTS=128, EXTRA_OFFSET=0     (full head: Q|K|V)
//   yolo_m9_qk_pack_i8_i8  : SLOTS=64,  EXTRA_OFFSET=0     (Q+K prefix only)
//   yolo_m9_v_pack_i8_i8   : SLOTS=64,  EXTRA_OFFSET=64    (V suffix only)
//
// All three share the m9 PSA call-site constants: in_w=16, twoc=256,
// head_stride=128, N=in_h*in_w=256.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_PACK_SLOTS
#error "YOLO_M9_PACK_SLOTS must be defined at compile time (e.g. 64 or 128)"
#endif
#ifndef YOLO_M9_PACK_EXTRA_OFFSET
#define YOLO_M9_PACK_EXTRA_OFFSET 0
#endif
#ifndef YOLO_M9_PACK_SYMBOL
#error "YOLO_M9_PACK_SYMBOL must be defined to the extern \"C\" symbol name"
#endif

// Hardcoded m9 call-site constants.
static constexpr int kInW = 16;
static constexpr int kTwoC = 256;
static constexpr int kHeadStride = 128;
static constexpr int kN = 256;
static constexpr int kSlots = YOLO_M9_PACK_SLOTS;
static constexpr int kExtraOffset = YOLO_M9_PACK_EXTRA_OFFSET;

extern "C" {

void YOLO_M9_PACK_SYMBOL(
    int8_t *in_row,  // (in_w, twoc) natural-layout qkv row
    int8_t *dst,     // (kSlots, kN) destination per-head buffer
    const int32_t /*input_width*/, const int32_t /*twoc*/,
    const int32_t /*slots*/, const int32_t /*head_stride*/,
    const int32_t /*N*/,
    const int32_t head_idx,    // 0 or 1
    const int32_t row_idx) {   // 0..in_h-1
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t chan_offset = head_idx * kHeadStride + kExtraOffset;
  const int32_t n_base = row_idx * kInW;

  AIE_LOOP_RANGE(kSlots, kSlots)
  for (int s = 0; s < kSlots; s++) {
    AIE_LOOP_RANGE(kInW, kInW)
    for (int x = 0; x < kInW; x++) {
      dst[s * kN + (n_base + x)] = in_row[x * kTwoC + (chan_offset + s)];
    }
  }

  event1();
}

} // extern "C"

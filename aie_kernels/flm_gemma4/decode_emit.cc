//===- decode_emit.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "decode_geometry.h"
#include "utils.h"

// Normalize the producers' stream granularity at the core. Every input to
// the projection memtile's temporal merge is one 512-element object.
#if EMIT_ATTENTION
using input_type = float; // finish leaves bf16 values in the float allocation.
#else
using input_type = bf16;
#endif
extern "C" void decode_emit(input_type *storage, bf16 *output, int chunk) {
  bf16 *input = reinterpret_cast<bf16 *>(storage);
#if EMIT_ATTENTION
  // The attention accumulator is [dh/8, head, 8]; projection wants [head, dh].
  // Assemble two groups into each aligned 256-bit output store.
  for (int i = 0; i < 512; i += 16) {
    int offset = chunk * 512 + i;
    int src =
        (offset / EMIT_DH) * 8 + (offset % EMIT_DH) / 8 * NUM_ATTN_HEADS * 8;
    auto lo = aie::load_v<8>(input + src);
    auto hi = aie::load_v<8>(input + src + NUM_ATTN_HEADS * 8);
    aie::store_v(output + i, aie::concat(lo, hi));
  }
#else
  copy_vectorized<bf16, 512>(output, input + 16 + chunk * 512);
#endif
}

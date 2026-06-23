//===- llama_bcast.cc ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Kernels for the broadcast probe (aie2_bcast_probe.py): one streamed table
// fanned out to two consumers. Proves one shim fill can feed both the lm_head
// GEMM and the embed-gather (the persistent loop's shim-saving primitive).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#ifndef LLAMA_BC_TILE
#define LLAMA_BC_TILE 64
#endif
#ifndef LLAMA_BC_SELECT
#define LLAMA_BC_SELECT 3
#endif

static constexpr int kTile = LLAMA_BC_TILE;
static constexpr int kSelect = LLAMA_BC_SELECT;

extern "C" {

// Reduce (lm_head-like): accumulate the sum of every streamed tile into out[0].
// Resets at tile_idx==0 so the running total is per-dispatch.
void bcast_reduce(int32_t *restrict tile, int32_t *restrict out,
                  int32_t tile_idx) {
  int32_t acc = (tile_idx == 0) ? 0 : out[0];
  for (int i = 0; i < kTile; i++)
    acc += tile[i];
  out[0] = acc;
}

// Select (gather-like): copy the kSelect-th streamed tile to out.
void bcast_select(int32_t *restrict tile, int32_t *restrict out,
                  int32_t tile_idx) {
  if (tile_idx != kSelect)
    return;
  for (int i = 0; i < kTile; i++)
    out[i] = tile[i];
}

} // extern "C"

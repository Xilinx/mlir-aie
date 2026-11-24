// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-lower-dma-channel-reset --verify-diagnostics %s -o /dev/null

// A hand-written re-arm binding cannot bypass the high-level rejection.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %b = aie.buffer(%tile) : memref<16xi32>
    %lock = aie.lock(%tile, 0) {init = 1 : i32}
    aie.mem(%tile) {
      aie.dma_start(MM2S, 0, ^bd, ^end) {core_pushes}
    ^bd:
      aie.dma_bd(%b : memref<16xi32>) {bd_id = 0 : i32}
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    aie.objectfifo_rearm_binding @r channels(%tile : index) locks(%lock : index) {channel_dirs = array<i32: 1>, channel_indices = array<i32: 0>, lock_inits = array<i32: 1>, head_bd_ids = array<i32: 0>, repeat_counts = array<i32: 0>}
    aie.runtime_sequence() {
      // expected-error @+1 {{does not support core-pushed DMA channels}}
      aiex.dma_channel_reset_for(@r)
    }
  }
}

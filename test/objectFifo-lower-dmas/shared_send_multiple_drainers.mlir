// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-lower-dmas --verify-diagnostics %s -o /dev/null

// Both segments must be sent before the producer can reuse the object.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<16xi32>
    aie.objectfifo.pool @p(%tile) {depth = 1 : i32, buffers = [@buffer]} : memref<16xi32> {
      aie.objectfifo.segment @lo {offset = 0 : i32, size = 8 : i32}
      aie.objectfifo.segment @hi {offset = 8 : i32, size = 8 : i32}
    }
    aie.objectfifo.core_endpoint @fill(%tile) fills @p {segments = [@lo, @hi]}
    // expected-note @+1 {{first core-pushed drainer is here}}
    aie.objectfifo.dma_endpoint @send_lo(%tile) drains @p {segments = [@lo], channelIndex = 0 : i32, release_strategy = #aie.release_strategy<push_queue>}
    // expected-error @+1 {{a pool supports only one core-pushed DMA drainer}}
    aie.objectfifo.dma_endpoint @send_hi(%tile) drains @p {segments = [@hi], channelIndex = 1 : i32, release_strategy = #aie.release_strategy<push_queue>}
    %core = aie.core(%tile) {
      %obj = aie.objectfifo.acquire @fill(1) : memref<16xi32>
      aie.objectfifo.release @fill(1)
      aie.end
    }
  }
}

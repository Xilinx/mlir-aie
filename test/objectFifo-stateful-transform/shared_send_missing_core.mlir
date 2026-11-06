// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-lower-dmas --verify-diagnostics %s

// A pool-level input must also have a core issuing its send tasks.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<16xi32>
    aie.objectfifo.pool @p(%tile) {depth = 1 : i32, buffers = [@buffer]} : memref<16xi32> {
      aie.objectfifo.segment @whole {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @fill(%tile) fills @p
    // expected-error @+1 {{push_queue requires producer releases in a local core to submit send tasks}}
    aie.objectfifo.dma_endpoint @send(%tile) drains @p {channelIndex = 0 : i32, release_strategy = #aie.release_strategy<push_queue>}
  }
}

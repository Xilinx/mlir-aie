// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-lower-dmas --verify-diagnostics %s -o /dev/null

// A release from an unrelated segment cannot submit the sender's tasks.
module @unrelated_release {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<16xi32>
    %free_lo = aie.lock(%tile) {sym_name = "free_lo", init = 1 : i32}
    %full_lo = aie.lock(%tile) {sym_name = "full_lo", init = 0 : i32}
    %free_hi = aie.lock(%tile) {sym_name = "free_hi", init = 1 : i32}
    %full_hi = aie.lock(%tile) {sym_name = "full_hi", init = 0 : i32}
    aie.objectfifo.pool @p(%tile) {depth = 1 : i32, buffers = [@buffer]} : memref<16xi32> {
      aie.objectfifo.segment @lo {offset = 0 : i32, size = 8 : i32, produceLock = @free_lo, consumeLock = @full_lo}
      aie.objectfifo.segment @hi {offset = 8 : i32, size = 8 : i32, produceLock = @free_hi, consumeLock = @full_hi}
    }
    aie.objectfifo.core_endpoint @fill_lo(%tile) fills @p {segments = [@lo]}
    aie.objectfifo.core_endpoint @fill_hi(%tile) fills @p {segments = [@hi]}
    // expected-error @+1 {{push_queue requires producer releases in a local core to submit send tasks}}
    aie.objectfifo.dma_endpoint @send_lo(%tile) drains @p {segments = [@lo], channelIndex = 0 : i32, release_strategy = #aie.release_strategy<push_queue>}
    aie.objectfifo.dma_endpoint @send_hi(%tile) drains @p {segments = [@hi], channelIndex = 1 : i32}
    %core = aie.core(%tile) {
      %lo = aie.objectfifo.acquire @fill_lo(1) : memref<8xi32, strided<[1]>>
      %hi = aie.objectfifo.acquire @fill_hi(1) : memref<8xi32, strided<[1], offset: 8>>
      aie.objectfifo.release @fill_hi(1)
      aie.end
    }
  }
}

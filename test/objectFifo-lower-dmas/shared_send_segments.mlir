// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s --check-prefix=DMA
// RUN: aie-opt --aie-objectfifo-lower-dmas --aie-objectfifo-lower-cores --canonicalize --aie-assign-bd-ids %s | FileCheck %s --check-prefix=CORE

// The cyclic drainer must not inherit the pushed drainer's tag.
// DMA: send_producer = @fill_lo
// DMA: aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 8) {bd_tag = 0 : i32}
// DMA: aie.dma_bd(%buffer : memref<16xi32> offset = 8 len = 8)
// DMA-NOT: bd_tag

// Only releasing the segment sent through push_queue submits a task.
// CORE: aie.core
// CORE: aie.use_lock(%full_lo, Release
// CORE: aie.dma_push_queue(MM2S : 0) bd_id
// CORE-NOT: aie.dma_push_queue
// CORE: aie.use_lock(%full_hi, Release
// CORE-NOT: aie.dma_push_queue
// CORE: aie.end

module @separate_fillers {
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
    aie.objectfifo.dma_endpoint @send_lo(%tile) drains @p {segments = [@lo], channelIndex = 0 : i32, release_strategy = #aie.release_strategy<push_queue>}
    aie.objectfifo.dma_endpoint @send_hi(%tile) drains @p {segments = [@hi], channelIndex = 1 : i32}
    %core = aie.core(%tile) {
      %lo = aie.objectfifo.acquire @fill_lo(1) : memref<8xi32, strided<[1]>>
      aie.objectfifo.release @fill_lo(1)
      %hi = aie.objectfifo.acquire @fill_hi(1) : memref<8xi32, strided<[1], offset: 8>>
      aie.objectfifo.release @fill_hi(1)
      aie.end
    }
  }
}

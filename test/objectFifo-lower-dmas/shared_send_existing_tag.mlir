// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-lower-dmas %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-lower-dmas --aie-objectfifo-lower-cores --canonicalize --aie-assign-bd-ids %s -o /dev/null

// Generated send tags must avoid tags on existing descriptors.
// CHECK: send_tags = array<i32: 1>
// CHECK: aie.dma_bd(%buffer : memref<16xi32>) {bd_tag = 0 : i32}
// CHECK: aie.dma_bd(%buffer : memref<16xi32> offset = 0 len = 16) {bd_tag = 1 : i32}
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %buffer = aie.buffer(%tile) {sym_name = "buffer"} : memref<16xi32>
    aie.objectfifo.pool @p(%tile) {depth = 1 : i32, buffers = [@buffer]} : memref<16xi32> {
      aie.objectfifo.segment @all {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @fill(%tile) fills @p
    aie.objectfifo.dma_endpoint @send(%tile) drains @p {channelIndex = 0 : i32, release_strategy = #aie.release_strategy<push_queue>}
    aie.mem(%tile) {
      aie.dma_start(MM2S, 1, ^bd, ^end)
    ^bd:
      aie.dma_bd(%buffer : memref<16xi32>) {bd_tag = 0 : i32}
      aie.next_bd ^bd
    ^end:
      aie.end
    }
    %core = aie.core(%tile) {
      %obj = aie.objectfifo.acquire @fill(1) : memref<16xi32>
      aie.objectfifo.release @fill(1)
      aie.end
    }
  }
}

// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Partial coordinates on both sides still allow a reachable neighbor.
module @possible_memtile_neighbors {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(1, ?)
    %reader = aie.logical_tile<MemTile>(0, ?)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p
  }
}
// CHECK-LABEL: module @possible_memtile_neighbors
// CHECK: @reader({{.*}}) drains @p {channelIndex = 0 : i32}

// -----

// Not every compatible column shares memory, but at least one does.
module @possible_core_neighbors {
  aie.device(npu2) {
    %home = aie.tile(0, 3)
    %reader = aie.logical_tile<CoreTile>(?, 2)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p
  }
}
// CHECK-LABEL: module @possible_core_neighbors
// CHECK: @reader({{.*}}) drains @p {channelIndex = 0 : i32}

// -----

// A DMA using local buffers can still use neighbor locks on channels 0-3.
module @possible_neighbor_locks {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(0, ?)
    %remote = aie.logical_tile<MemTile>(1, ?)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    %free = aie.lock(%remote) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%remote) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {
        offset = 0 : i32, size = 16 : i32,
        produceLock = @free, consumeLock = @full
      }
    }
    aie.objectfifo.dma_endpoint @reader(%home) drains @p {channelIndex = 3 : i32}
  }
}
// CHECK-LABEL: module @possible_neighbor_locks
// CHECK: @reader({{.*}}) drains @p {channelIndex = 3 : i32}

// -----

// Compute DMA locks must be local, but partial coordinates can still resolve
// to the fixed lock module's tile. Do not require SSA identity before placement.
module @possible_local_core_dma_locks {
  aie.device(npu2) {
    %home = aie.tile(0, 2)
    %reader = aie.logical_tile<CoreTile>(0, ?)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    %free = aie.lock(%home) {sym_name = "free", init = 1 : i32}
    %full = aie.lock(%home) {sym_name = "full", init = 0 : i32}
    aie.objectfifo.pool @p(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {
        offset = 0 : i32, size = 16 : i32,
        produceLock = @free, consumeLock = @full
      }
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @p
  }
}
// CHECK-LABEL: module @possible_local_core_dma_locks
// CHECK: @reader({{.*}}) drains @p {channelIndex = 0 : i32}

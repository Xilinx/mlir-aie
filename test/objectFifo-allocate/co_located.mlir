// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-place-tiles --aie-objectfifo-lower-dmas --aie-assign-lock-ids --aie-assign-buffer-addresses %s -o /dev/null
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Known equal coordinates guarantee locality. Placement resolves the two SSA
// tile references before the DMA verifier requires a canonical tile operand.
module {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %reader = aie.logical_tile<MemTile>(0, 1)
    aie.objectfifo.pool @shared(%home) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @shared {channelIndex = 5 : i32}
    aie.objectfifo.dma_endpoint @home_reader(%home) drains @shared
    aie.objectfifo.dma_endpoint @alias_reader(%reader) drains @shared
    aie.objectfifo.dma_endpoint @home_writer(%home) fills @shared
    aie.objectfifo.dma_endpoint @alias_writer(%reader) fills @shared
  }
}
// CHECK: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK: aie.buffer(%[[HOME]]) {sym_name = "shared_buff_0"}
// CHECK: @reader({{.*}}) drains @shared {channelIndex = 5 : i32}
// CHECK: @home_reader({{.*}}) drains @shared {channelIndex = 0 : i32}
// CHECK: @alias_reader({{.*}}) drains @shared {channelIndex = 1 : i32}
// CHECK: @home_writer({{.*}}) fills @shared {channelIndex = 0 : i32}
// CHECK: @alias_writer({{.*}}) fills @shared {channelIndex = 1 : i32}

// -----

// Reservations from existing DMA programs apply to fully resolved logical
// aliases too, independently in each direction.
module @existing_dma {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %alias = aie.logical_tile<MemTile>(0, 1)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.memtile_dma(%home) {
      aie.dma_start(MM2S, 0, ^read, ^next)
    ^next:
      aie.dma_start(S2MM, 0, ^write, ^end)
    ^read:
      aie.dma_bd(%b : memref<16xi32>, 0, 16)
      aie.next_bd ^read
    ^write:
      aie.dma_bd(%b : memref<16xi32>, 0, 16)
      aie.next_bd ^write
    ^end:
      aie.end
    }
    aie.objectfifo.pool @p(%alias) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%alias) drains @p
    aie.objectfifo.dma_endpoint @writer(%alias) fills @p
  }
}
// CHECK-LABEL: module @existing_dma
// CHECK: @reader({{.*}}) drains @p {channelIndex = 1 : i32}
// CHECK: @writer({{.*}}) fills @p {channelIndex = 1 : i32}

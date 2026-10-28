//===- logical_tiles.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Pools and endpoints name a tile through the TileLike interface, so a design
// whose tiles are not placed yet lowers just as far. Placement may run after
// the objectFifo pipeline.

// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform="skip-verify=true" --aie-place-tiles %s | FileCheck %s --check-prefix=PLACED

module {
  aie.device(xcve2302) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)

    aie.objectfifo.pool @p(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // A shim end's objects live in DDR; the pool names none of its own.
    aie.objectfifo.pool @shim_p(%shim) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @src(%shim) drains @shim_p {fifoName = "of"}
    aie.objectfifo.dma_endpoint @dst(%mem) fills @p {fifoName = "of"}
    aie.route from @src to [@dst]
  }
}

// Buffers, locks and channels are all assigned against the unplaced tiles.
// CHECK-DAG:   %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK-DAG:   %[[MEM:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK:       aie.buffer(%[[MEM]]) {sym_name = "p_buff_0"}
// CHECK:       aie.buffer(%[[MEM]]) {sym_name = "p_buff_1"}
// CHECK:       aie.lock(%[[MEM]]) {init = 2 : i32, sym_name = "p_prod_lock_0"}
// CHECK:       aie.lock(%[[MEM]]) {init = 0 : i32, sym_name = "p_cons_lock_0"}
// CHECK:       aie.objectfifo.dma_endpoint @src(%[[SHIM]]) drains @shim_p {channelIndex = 0 : i32
// CHECK:       aie.objectfifo.dma_endpoint @dst(%[[MEM]]) fills @p {channelIndex = 0 : i32
// CHECK:       aie.flow(%[[SHIM]], DMA : 0, %[[MEM]], DMA : 0)
// CHECK:       aie.shim_dma_allocation @of_shim_alloc(%[[SHIM]], MM2S, 0)

// Placing afterwards leaves ordinary physical IR behind.
// PLACED-DAG:  %[[PMEM:.*]] = aie.tile(2, 1)
// PLACED-DAG:  %[[PSHIM:.*]] = aie.tile(2, 0)
// PLACED-NOT:  aie.logical_tile
// PLACED:      aie.flow(%[[PSHIM]], DMA : 0, %[[PMEM]], DMA : 0)
// PLACED:      aie.memtile_dma(%[[PMEM]])
// PLACED:      aie.dma_start(S2MM, 0

// -----

// Known coordinates on logical tiles have the same memory affinity as their
// physical equivalents, even before placement replaces the tile operations.
module @shared_logical {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(0, 1)
    %reader = aie.logical_tile<MemTile>(1, 1)
    aie.objectfifo.pool @shared(%home) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%reader) drains @shared
  }
}
// CHECK-LABEL: module @shared_logical
// CHECK: %[[HOME:.*]] = aie.logical_tile<MemTile>(0, 1)
// CHECK: aie.buffer(%[[HOME]]) {sym_name = "shared_buff_0"}
// CHECK: aie.buffer(%[[HOME]]) {sym_name = "shared_buff_1"}
// CHECK: @reader({{.*}}) drains @shared {channelIndex = 0 : i32}
// PLACED-LABEL: module @shared_logical
// PLACED: %[[PHOME:.*]] = aie.tile(0, 1)
// PLACED: aie.buffer(%[[PHOME]]) {sym_name = "shared_buff_0"}
// PLACED: aie.dma_start(MM2S, 0

// -----

// Unknown coordinates do not reduce the capacity of proven-local accesses or
// alias two independent logical tiles just because their constraints match.
module @local_unplaced {
  aie.device(npu2) {
    %a = aie.logical_tile<MemTile>(?, ?)
    %b = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo.pool @a(%a) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @b(%b) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @a0(%a) drains @a
    aie.objectfifo.dma_endpoint @a1(%a) drains @a
    aie.objectfifo.dma_endpoint @a2(%a) drains @a
    aie.objectfifo.dma_endpoint @a3(%a) drains @a
    aie.objectfifo.dma_endpoint @a4(%a) drains @a
    aie.objectfifo.dma_endpoint @a5(%a) drains @a
    aie.objectfifo.dma_endpoint @b0(%b) drains @b
    aie.objectfifo.dma_endpoint @b5(%b) fills @b {channelIndex = 5 : i32}
  }
}
// CHECK-LABEL: module @local_unplaced
// CHECK: @a0({{.*}}) drains @a {channelIndex = 0 : i32}
// CHECK: @a1({{.*}}) drains @a {channelIndex = 1 : i32}
// CHECK: @a2({{.*}}) drains @a {channelIndex = 2 : i32}
// CHECK: @a3({{.*}}) drains @a {channelIndex = 3 : i32}
// CHECK: @a4({{.*}}) drains @a {channelIndex = 4 : i32}
// CHECK: @a5({{.*}}) drains @a {channelIndex = 5 : i32}
// CHECK: @b0({{.*}}) drains @b {channelIndex = 0 : i32}
// CHECK: @b5({{.*}}) fills @b {channelIndex = 5 : i32}
// PLACED-LABEL: module @local_unplaced
// PLACED-NOT: aie.logical_tile
// PLACED: aie.dma_start(MM2S, 5

// -----

// Four remote transfers and two local transfers still fit before placement.
// Fix the remote buffers so success cannot rely on relocating the pool.
module @remote_unplaced {
  aie.device(npu2) {
    %home = aie.logical_tile<MemTile>(1, 1)
    %reader = aie.logical_tile<MemTile>(0, ?)
    %b = aie.buffer(%home) {sym_name = "b"} : memref<16xi32>
    aie.objectfifo.pool @remote(%home) {depth = 1 : i32, buffers = [@b]} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.pool @local(%reader) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @local0(%reader) drains @local
    aie.objectfifo.dma_endpoint @local1(%reader) drains @local
    aie.objectfifo.dma_endpoint @remote0(%reader) drains @remote
    aie.objectfifo.dma_endpoint @remote1(%reader) drains @remote
    aie.objectfifo.dma_endpoint @remote2(%reader) drains @remote
    aie.objectfifo.dma_endpoint @remote3(%reader) drains @remote
  }
}
// CHECK-LABEL: module @remote_unplaced
// CHECK: @local0({{.*}}) drains @local {channelIndex = 4 : i32}
// CHECK: @local1({{.*}}) drains @local {channelIndex = 5 : i32}
// CHECK: @remote0({{.*}}) drains @remote {channelIndex = 0 : i32}
// CHECK: @remote1({{.*}}) drains @remote {channelIndex = 1 : i32}
// CHECK: @remote2({{.*}}) drains @remote {channelIndex = 2 : i32}
// CHECK: @remote3({{.*}}) drains @remote {channelIndex = 3 : i32}
// PLACED-LABEL: module @remote_unplaced
// PLACED-NOT: aie.logical_tile
// PLACED: aie.dma_start(MM2S, 5

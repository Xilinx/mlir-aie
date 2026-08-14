//===- logical_tiles.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Pools and endpoints name a tile through the TileLike interface, so a design
// whose tiles are not placed yet lowers just as far. Placement can then run
// after the objectFifo pipeline rather than before it.

// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-place-tiles %s | FileCheck %s --check-prefix=PLACED

module {
  aie.device(xcve2302) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)

    aie.objectfifo.pool @p(%mem) {depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
    aie.objectfifo.dma_endpoint @src(%shim) {fifoName = "of"}
    aie.objectfifo.dma_endpoint @dst(%mem) fills @p {fifoName = "of"}
    aie.objectfifo.flow from @src to [@dst]
  }
}

// Buffers, locks and channels are all assigned against the unplaced tiles.
// CHECK-DAG:   %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(?, ?)
// CHECK-DAG:   %[[MEM:.*]] = aie.logical_tile<MemTile>(?, ?)
// CHECK:       aie.buffer(%[[MEM]]) {sym_name = "p_buff_0"}
// CHECK:       aie.buffer(%[[MEM]]) {sym_name = "p_buff_1"}
// CHECK:       aie.lock(%[[MEM]]) {init = 2 : i32, sym_name = "p_prod_lock_0"}
// CHECK:       aie.lock(%[[MEM]]) {init = 0 : i32, sym_name = "p_cons_lock_0"}
// CHECK:       aie.objectfifo.dma_endpoint @src(%[[SHIM]]) {channel = #aie.objectfifo_channel<MM2S : 0>
// CHECK:       aie.objectfifo.dma_endpoint @dst(%[[MEM]]) fills @p {channel = #aie.objectfifo_channel<S2MM : 0>
// CHECK:       aie.flow(%[[SHIM]], DMA : 0, %[[MEM]], DMA : 0)
// CHECK:       aie.shim_dma_allocation @of_shim_alloc(%[[SHIM]], MM2S, 0)

// Placing afterwards leaves ordinary physical IR behind.
// PLACED-DAG:  %[[PMEM:.*]] = aie.tile(2, 1)
// PLACED-DAG:  %[[PSHIM:.*]] = aie.tile(2, 0)
// PLACED-NOT:  aie.logical_tile
// PLACED:      aie.flow(%[[PSHIM]], DMA : 0, %[[PMEM]], DMA : 0)
// PLACED:      aie.memtile_dma(%[[PMEM]])
// PLACED:      aie.dma_start(S2MM, 0

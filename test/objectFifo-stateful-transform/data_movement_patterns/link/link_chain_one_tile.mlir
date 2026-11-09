//===- link_chain_one_tile.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Two chained links on one memtile: @mid leaves the tile and comes back in
// through its stream switch.

// CHECK-DAG: aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
// CHECK-DAG: aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 2)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 1, %mem_tile_1_1, DMA : 1)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
module @chain_on_one_tile {
  aie.device(npu2) {
    %shim = aie.tile(1, 0)
    %mem11 = aie.tile(1, 1)
    aie.objectfifo @a (%shim, {%mem11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%shim, {%mem11}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%mem11, {%mem11}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%mem11, {%shim}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@mid] ([0, 16] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}

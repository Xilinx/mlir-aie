//===- link_chain_join.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A join of joins: two memtiles each join two cores' slots, and a third joins
// their outputs, @left and @right, each one link's output and the next's
// input. Each moves from its own link's pool, walked segment by segment, into
// its slot of the last link's pool.

// CHECK-DAG: %[[LEFT0:.*]] = aie.buffer(%mem_tile_0_1) {sym_name = "left__buf0"} : memref<32xi32>
// CHECK-DAG: %[[RIGHT0:.*]] = aie.buffer(%mem_tile_2_1) {sym_name = "right__buf0"} : memref<32xi32>
// CHECK-DAG: %[[OUT0:.*]] = aie.buffer(%mem_tile_1_1) {sym_name = "out__buf0"} : memref<64xi32>
// CHECK-DAG: aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 1)
// CHECK-DAG: aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
// CHECK-DAG: aie.flow(%tile_2_3, DMA : 0, %mem_tile_2_1, DMA : 1)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %mem_tile_1_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_2_1, DMA : 0, %mem_tile_1_1, DMA : 1)
// CHECK-DAG: aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)

// The first joins fill their slots and send both, in order.
// CHECK-LABEL: aie.memtile_dma(%mem_tile_0_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_bd(%[[LEFT0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.dma_bd(%[[LEFT0]] : memref<32xi32> offset = 16 len = 16)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%[[LEFT0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[LEFT0]] : memref<32xi32> offset = 16 len = 16)

// CHECK-LABEL: aie.memtile_dma(%mem_tile_2_1)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%[[RIGHT0]] : memref<32xi32> offset = 0 len = 16)
// CHECK:       aie.dma_bd(%[[RIGHT0]] : memref<32xi32> offset = 16 len = 16)

// The last join takes each of them whole into its slot.
// CHECK-LABEL: aie.memtile_dma(%mem_tile_1_1)
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_bd(%[[OUT0]] : memref<64xi32> offset = 0 len = 32)
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.dma_bd(%[[OUT0]] : memref<64xi32> offset = 32 len = 32)
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%[[OUT0]] : memref<64xi32> offset = 0 len = 32)
// CHECK:       aie.dma_bd(%[[OUT0]] : memref<64xi32> offset = 32 len = 32)

module @join_of_joins {
  aie.device(npu2) {
    %shim = aie.tile(1, 0)
    %mem01 = aie.tile(0, 1)
    %mem11 = aie.tile(1, 1)
    %mem21 = aie.tile(2, 1)
    %core02 = aie.tile(0, 2)
    %core03 = aie.tile(0, 3)
    %core22 = aie.tile(2, 2)
    %core23 = aie.tile(2, 3)
    aie.objectfifo @a (%core02, {%mem01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%core03, {%mem01}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c (%core22, {%mem21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d (%core23, {%mem21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @left (%mem01, {%mem11}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @right (%mem21, {%mem11}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%mem11, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a, @b] -> [@left] ([0, 16] [])
    aie.objectfifo.link [@c, @d] -> [@right] ([0, 16] [])
    aie.objectfifo.link [@left, @right] -> [@out] ([0, 32] [])
    aie.core(%core02) {
      %e = aie.objectfifo.acquire @a (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a (Produce, 1)
      aie.end
    }
    aie.core(%core03) {
      %e = aie.objectfifo.acquire @b (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b (Produce, 1)
      aie.end
    }
    aie.core(%core22) {
      %e = aie.objectfifo.acquire @c (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c (Produce, 1)
      aie.end
    }
    aie.core(%core23) {
      %e = aie.objectfifo.acquire @d (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @d (Produce, 1)
      aie.end
    }
  }
}

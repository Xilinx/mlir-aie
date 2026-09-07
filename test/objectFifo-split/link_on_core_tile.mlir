//===- link_on_core_tile.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify %s | FileCheck %s

// A link point on a compute tile partitions its pool into one segment per
// participant, the same way a memtile does. The tile's own core takes no part:
// both ends of the link are DMA endpoints.

module @distribute_on_core {
  aie.device(npu1) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile04 = aie.tile(0, 4)
    %tile05 = aie.tile(0, 5)

    aie.objectfifo @in (%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out0 (%tile02, {%tile04}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1 (%tile02, {%tile05}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out0, @out1] ([][0, 16])

    %core04 = aie.core(%tile04) {
      %e = aie.objectfifo.acquire @out0 (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out0 (Consume, 1)
      aie.end
    }
    %core05 = aie.core(%tile05) {
      %e = aie.objectfifo.acquire @out1 (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out1 (Consume, 1)
      aie.end
    }
  }
}

// CHECK-LABEL: @distribute_on_core
// CHECK-DAG:   %[[T00:.*]] = aie.tile(0, 0)
// CHECK-DAG:   %[[T02:.*]] = aie.tile(0, 2)
// CHECK-DAG:   %[[T04:.*]] = aie.tile(0, 4)
// CHECK-DAG:   %[[T05:.*]] = aie.tile(0, 5)

// CHECK:   aie.objectfifo.pool @in_cons_pool(%[[T02]]) {depth = 2 : i32, fifoName = "in"} : memref<32xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
// CHECK: aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
// CHECK:   aie.route_endpoint @in_prod_dma(%[[T00]]) DMA {fifoName = "in"}
// CHECK:   aie.objectfifo.dma_endpoint @in_cons_dma(%[[T02]]) fills @in_cons_pool {fifoName = "in", segments = [@s0, @s1]}

// Each output drains its own segment of the shared object.
// CHECK:   aie.objectfifo.dma_endpoint @out0_prod_dma(%[[T02]]) drains @in_cons_pool {fifoName = "out0", segments = [@s0]}
// CHECK:   aie.objectfifo.dma_endpoint @out1_prod_dma(%[[T02]]) drains @in_cons_pool {fifoName = "out1", segments = [@s1]}
// CHECK-NOT: aie.objectfifo.link

// -----

module @join_on_core {
  aie.device(npu1) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile04 = aie.tile(0, 4)
    %tile05 = aie.tile(0, 5)

    aie.objectfifo @in0 (%tile04, {%tile02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1 (%tile05, {%tile02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out (%tile02, {%tile00}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 16][])

    %core04 = aie.core(%tile04) {
      %e = aie.objectfifo.acquire @in0 (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in0 (Produce, 1)
      aie.end
    }
    %core05 = aie.core(%tile05) {
      %e = aie.objectfifo.acquire @in1 (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in1 (Produce, 1)
      aie.end
    }
  }
}

// CHECK-LABEL: @join_on_core
// CHECK-DAG:   %[[U00:.*]] = aie.tile(0, 0)
// CHECK-DAG:   %[[U02:.*]] = aie.tile(0, 2)

// Each input fills its own segment of the shared object.
// CHECK:   aie.objectfifo.dma_endpoint @in0_cons_dma(%[[U02]]) fills @out_pool {fifoName = "in0", segments = [@s0]}
// CHECK:   aie.objectfifo.dma_endpoint @in1_cons_dma(%[[U02]]) fills @out_pool {fifoName = "in1", segments = [@s1]}

// CHECK:   aie.objectfifo.pool @out_pool(%[[U02]]) {depth = 2 : i32, fifoName = "out"} : memref<32xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
// CHECK: aie.objectfifo.segment @s1 {offset = 16 : i32, size = 16 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @out_prod_dma(%[[U02]]) drains @out_pool {fifoName = "out", segments = [@s0, @s1]}
// CHECK:   aie.route_endpoint @out_cons_dma(%[[U00]]) DMA {fifoName = "out"}
// CHECK-NOT: aie.objectfifo.link

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A join gives the memtile one pool whose segments are the pieces the inputs
// write; each input's memtile-side DMA endpoint selects its own segment, while
// the output's endpoint drains all of them.

module @link_join {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @in0 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 16][])
  }
}

// CHECK-LABEL: @link_join
// CHECK-DAG:   %[[T20:.*]] = aie.tile(2, 0)
// CHECK-DAG:   %[[T21:.*]] = aie.tile(2, 1)
// CHECK-DAG:   %[[T22:.*]] = aie.tile(2, 2)
// CHECK-DAG:   %[[T23:.*]] = aie.tile(2, 3)

// Each input keeps a whole-object pool on its own tile.
// CHECK:   aie.objectfifo.pool @in0_pool(%[[T22]]) {depth = 2 : i32, fifoName = "in0", segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
// CHECK:   aie.objectfifo.dma_endpoint @in0_prod_dma(%[[T22]]) drains @in0_pool
// CHECK:   aie.objectfifo.dma_endpoint @in0_cons_dma(%[[T21]]) fills @out_pool {fifoName = "in0", segments = array<i32: 0>}
// CHECK:   aie.objectfifo.flow from @in0_prod_dma to [@in0_cons_dma]
// CHECK:   aie.objectfifo.pool @in1_pool(%[[T23]]) {depth = 2 : i32, fifoName = "in1", segments = [#aie.objectfifo_segment<offset = 0, size = 32>]} : memref<32xi32>
// CHECK:   aie.objectfifo.dma_endpoint @in1_prod_dma(%[[T23]]) drains @in1_pool
// CHECK:   aie.objectfifo.dma_endpoint @in1_cons_dma(%[[T21]]) fills @out_pool {fifoName = "in1", segments = array<i32: 1>}
// CHECK:   aie.objectfifo.flow from @in1_prod_dma to [@in1_cons_dma]

// The memtile holds one pool whose segments are the pieces the inputs write.
// CHECK:   aie.objectfifo.pool @out_pool(%[[T21]]) {depth = 2 : i32, fifoName = "out", segments = [#aie.objectfifo_segment<offset = 0, size = 16>, #aie.objectfifo_segment<offset = 16, size = 32>]} : memref<48xi32>
// CHECK:   aie.objectfifo.dma_endpoint @out_prod_dma(%[[T21]]) drains @out_pool {fifoName = "out", segments = array<i32: 0, 1>}
// CHECK:   aie.objectfifo.dangling_endpoint @out_cons_dma(%[[T20]]) S2MM DMA {fifoName = "out"}
// CHECK:   aie.objectfifo.flow from @out_prod_dma to [@out_cons_dma]
// CHECK-NOT: aie.objectfifo.link

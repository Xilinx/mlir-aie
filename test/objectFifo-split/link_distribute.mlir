// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A distribute is the mirror of a join: the input fills the whole memtile pool,
// and each output drains one segment of it.

module @link_distribute {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @in (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @out0 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<32xi32>>

    aie.objectfifo.link [@in] -> [@out0, @out1] ([][0, 16])
  }
}

// CHECK-LABEL: @link_distribute
// CHECK-DAG:   %[[T20:.*]] = aie.tile(2, 0)
// CHECK-DAG:   %[[T21:.*]] = aie.tile(2, 1)
// CHECK-DAG:   %[[T22:.*]] = aie.tile(2, 2)
// CHECK-DAG:   %[[T23:.*]] = aie.tile(2, 3)

// CHECK:   aie.objectfifo.pool @in_cons_pool(%[[T21]]) {depth = 2 : i32, fifoName = "in", segments = [#aie.objectfifo_segment<offset = 0, size = 16>, #aie.objectfifo_segment<offset = 16, size = 32>]} : memref<48xi32>
// CHECK:   aie.objectfifo.pool @in_pool(%[[T20]]) {depth = 2 : i32, fifoName = "in", segments = [#aie.objectfifo_segment<offset = 0, size = 48>]} : memref<48xi32>
// CHECK:   aie.objectfifo.dma_endpoint @in_prod_dma(%[[T20]]) drains @in_pool {fifoName = "in"}
// CHECK:   aie.objectfifo.dma_endpoint @in_cons_dma(%[[T21]]) fills @in_cons_pool {fifoName = "in"}
// CHECK:   aie.objectfifo.flow from @in_prod_dma to [@in_cons_dma]

// Each output drains one segment of the memtile pool.
// CHECK:   aie.objectfifo.dma_endpoint @out0_prod_dma(%[[T21]]) drains @in_cons_pool {fifoName = "out0", segments = array<i32: 0>}
// CHECK:   aie.objectfifo.pool @out0_cons_pool(%[[T22]]) {depth = 2 : i32, fifoName = "out0", segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
// CHECK:   aie.objectfifo.dma_endpoint @out0_cons_dma(%[[T22]]) fills @out0_cons_pool
// CHECK:   aie.objectfifo.flow from @out0_prod_dma to [@out0_cons_dma]
// CHECK:   aie.objectfifo.dma_endpoint @out1_prod_dma(%[[T21]]) drains @in_cons_pool {fifoName = "out1", segments = array<i32: 1>}
// CHECK:   aie.objectfifo.pool @out1_cons_pool(%[[T23]]) {depth = 2 : i32, fifoName = "out1", segments = [#aie.objectfifo_segment<offset = 0, size = 32>]} : memref<32xi32>
// CHECK:   aie.objectfifo.dma_endpoint @out1_cons_dma(%[[T23]]) fills @out1_cons_pool
// CHECK:   aie.objectfifo.flow from @out1_prod_dma to [@out1_cons_dma]

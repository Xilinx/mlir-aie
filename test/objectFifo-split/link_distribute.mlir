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

// CHECK:   aie.objectfifo.pool @in__cons__pool(%[[T21]]) {depth = 2 : i32, fifoName = "in"} : memref<48xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
// CHECK: aie.objectfifo.segment @s1 {offset = 16 : i32, size = 32 : i32}
// CHECK:   aie.route_endpoint @in__prod__dma(%[[T20]]) DMA {fifoName = "in"}
// CHECK:   aie.objectfifo.dma_endpoint @in__cons__dma(%[[T21]]) fills @in__cons__pool {fifoName = "in", segments = [@s0, @s1]}
// CHECK:   aie.route from @in__prod__dma to [@in__cons__dma]

// Each output drains one segment of the memtile pool.
// CHECK:   aie.objectfifo.dma_endpoint @out0__prod__dma(%[[T21]]) drains @in__cons__pool {fifoName = "out0", segments = [@s0]}
// CHECK:   aie.objectfifo.pool @out0__cons__pool(%[[T22]]) {depth = 2 : i32, fifoName = "out0"} : memref<16xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @out0__cons__dma(%[[T22]]) fills @out0__cons__pool
// CHECK:   aie.route from @out0__prod__dma to [@out0__cons__dma]
// CHECK:   aie.objectfifo.dma_endpoint @out1__prod__dma(%[[T21]]) drains @in__cons__pool {fifoName = "out1", segments = [@s1]}
// CHECK:   aie.objectfifo.pool @out1__cons__pool(%[[T23]]) {depth = 2 : i32, fifoName = "out1"} : memref<32xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @out1__cons__dma(%[[T23]]) fills @out1__cons__pool
// CHECK:   aie.route from @out1__prod__dma to [@out1__cons__dma]

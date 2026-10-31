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
// CHECK:   aie.objectfifo.pool @in0__pool(%[[T22]]) {depth = 2 : i32, fifoName = "in0"} : memref<16xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @in0__prod__dma(%[[T22]]) drains @in0__pool
// CHECK:   aie.objectfifo.dma_endpoint @in0__cons__dma(%[[T21]]) fills @out__pool {fifoName = "in0", segments = [@s0]}
// CHECK:   aie.route from @in0__prod__dma to [@in0__cons__dma]
// CHECK:   aie.objectfifo.pool @in1__pool(%[[T23]]) {depth = 2 : i32, fifoName = "in1"} : memref<32xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 32 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @in1__prod__dma(%[[T23]]) drains @in1__pool
// CHECK:   aie.objectfifo.dma_endpoint @in1__cons__dma(%[[T21]]) fills @out__pool {fifoName = "in1", segments = [@s1]}
// CHECK:   aie.route from @in1__prod__dma to [@in1__cons__dma]

// The memtile holds one pool whose segments are the pieces the inputs write.
// CHECK:   aie.objectfifo.pool @out__pool(%[[T21]]) {depth = 2 : i32, fifoName = "out"} : memref<48xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
// CHECK: aie.objectfifo.segment @s1 {offset = 16 : i32, size = 32 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @out__prod__dma(%[[T21]]) drains @out__pool {fifoName = "out", segments = [@s0, @s1]}
// CHECK:   aie.route_endpoint @out__cons__dma(%[[T20]]) DMA {fifoName = "out"}
// CHECK:   aie.route from @out__prod__dma to [@out__cons__dma]
// CHECK-NOT: aie.objectfifo.link

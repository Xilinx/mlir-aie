// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A fresh fifo added next to already-split IR is lowered on its own terms; the
// pools and endpoints that are already there are untouched.

module @incremental {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo.pool @done__pool(%tile12) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @done__prod(%tile12) fills @done__pool
    aie.objectfifo.core_endpoint @done__cons(%tile13) drains @done__pool

    aie.objectfifo @fresh (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  }
}

// CHECK-LABEL: @incremental
// CHECK:   aie.objectfifo.pool @done__pool
// CHECK:   aie.objectfifo.core_endpoint @done__prod
// CHECK:   aie.objectfifo.core_endpoint @done__cons
// CHECK:   aie.objectfifo.pool @fresh__pool({{.*}}) {depth = 2 : i32, fifoName = "fresh"} : memref<8xi32>
// CHECK: aie.objectfifo.segment @s0 {offset = 0 : i32, size = 8 : i32}
// CHECK:   aie.objectfifo.dma_endpoint @fresh__prod__dma({{.*}}) drains @fresh__pool
// CHECK:   aie.objectfifo.pool @fresh__cons__pool
// CHECK:   aie.objectfifo.dma_endpoint @fresh__cons__dma({{.*}}) fills @fresh__cons__pool
// CHECK:   aie.route from @fresh__prod__dma to [@fresh__cons__dma]

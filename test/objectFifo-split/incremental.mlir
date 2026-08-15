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

    aie.objectfifo.pool @done_pool(%tile12) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.core_endpoint @done_prod(%tile12) fills @done_pool
    aie.objectfifo.core_endpoint @done_cons(%tile13) drains @done_pool

    aie.objectfifo @fresh (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
  }
}

// CHECK-LABEL: @incremental
// CHECK:   aie.objectfifo.pool @done_pool
// CHECK:   aie.objectfifo.core_endpoint @done_prod
// CHECK:   aie.objectfifo.core_endpoint @done_cons
// CHECK:   aie.objectfifo.pool @fresh_pool({{.*}}) {depth = 2 : i32, fifoName = "fresh", segments = [#aie.objectfifo_segment<offset = 0, size = 8>]} : memref<8xi32>
// CHECK:   aie.objectfifo.dma_endpoint @fresh_prod_dma({{.*}}) drains @fresh_pool
// CHECK:   aie.objectfifo.pool @fresh_cons_pool
// CHECK:   aie.objectfifo.dma_endpoint @fresh_cons_dma({{.*}}) fills @fresh_cons_pool
// CHECK:   aie.objectfifo.flow from @fresh_prod_dma to [@fresh_cons_dma]

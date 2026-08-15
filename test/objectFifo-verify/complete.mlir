// RUN: aie-opt --aie-objectfifo-verify %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-verify %s -o %t1.mlir
// RUN: aie-opt --aie-objectfifo-verify %t1.mlir -o %t2.mlir
// RUN: diff %t1.mlir %t2.mlir

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A complete design: every segment is filled by one endpoint and drained by
// another, and every DMA endpoint is connected. Verification changes nothing.

module @complete {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo.pool @prod_pool(%tile12) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.core_endpoint @prod_core(%tile12) fills @prod_pool
    aie.objectfifo.dma_endpoint @prod_dma(%tile12) drains @prod_pool

    aie.objectfifo.pool @cons_pool(%tile33) {
      depth = 2 : i32, segments = [#aie.objectfifo_segment<offset = 0, size = 16>]
    } : memref<16xi32>
    aie.objectfifo.dma_endpoint @cons_dma(%tile33) fills @cons_pool
    aie.objectfifo.core_endpoint @cons_core(%tile33) drains @cons_pool

    aie.objectfifo.flow from @prod_dma to [@cons_dma]

    %core12 = aie.core(%tile12) {
      %elem = aie.objectfifo.acquire @prod_core : memref<16xi32>
      aie.objectfifo.release @prod_core [1]
      aie.end
    }
  }
}

// CHECK-LABEL: @complete
// CHECK: aie.objectfifo.pool @prod_pool
// CHECK: aie.objectfifo.flow from @prod_dma to [@cons_dma]

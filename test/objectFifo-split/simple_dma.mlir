// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Non-adjacent tiles: a pool at each end, a DMA endpoint at each end, and a
// flow between them.

module @simple_dma {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %elem = aie.objectfifo.acquire @of1 (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @of1 (Produce, 1)
      aie.end
    }

    %core33 = aie.core(%tile33) {
      %elem = aie.objectfifo.acquire @of1 (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @of1 (Consume, 1)
      aie.end
    }
  }
}

// CHECK-LABEL: @simple_dma
// CHECK-DAG:   %[[T12:.*]] = aie.tile(1, 2)
// CHECK-DAG:   %[[T33:.*]] = aie.tile(3, 3)
// CHECK:   aie.objectfifo.pool @of1_pool(%[[T12]]) {depth = 2 : i32, fifoName = "of1", segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
// CHECK:   aie.objectfifo.core_endpoint @of1_prod(%[[T12]]) fills @of1_pool
// CHECK:   aie.objectfifo.dma_endpoint @of1_prod_dma(%[[T12]]) drains @of1_pool {fifoName = "of1"}
// CHECK:   aie.objectfifo.pool @of1_cons_pool(%[[T33]]) {depth = 2 : i32, fifoName = "of1", segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
// CHECK:   aie.objectfifo.core_endpoint @of1_cons(%[[T33]]) drains @of1_cons_pool
// CHECK:   aie.objectfifo.dma_endpoint @of1_cons_dma(%[[T33]]) fills @of1_cons_pool {fifoName = "of1"}
// CHECK:   aie.objectfifo.flow from @of1_prod_dma to [@of1_cons_dma]
// CHECK-NOT: aie.objectfifo @

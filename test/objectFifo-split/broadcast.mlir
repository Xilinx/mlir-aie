// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One source channel reaches every consumer, so a broadcast becomes a single
// flow with several destinations; each consumer keeps its own pool.

module @broadcast {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @bc (%tile12, {%tile33, %tile13}, [2, 2, 3]) : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %elem = aie.objectfifo.acquire @bc (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @bc (Produce, 1)
      aie.end
    }

    %core33 = aie.core(%tile33) {
      %elem = aie.objectfifo.acquire @bc (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @bc (Consume, 1)
      aie.end
    }

    %core13 = aie.core(%tile13) {
      %elem = aie.objectfifo.acquire @bc (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @bc (Consume, 1)
      aie.end
    }
  }
}

// CHECK-LABEL: @broadcast
// CHECK-DAG:   %[[T12:.*]] = aie.tile(1, 2)
// CHECK-DAG:   %[[T33:.*]] = aie.tile(3, 3)
// CHECK-DAG:   %[[T13:.*]] = aie.tile(1, 3)
// CHECK:   aie.objectfifo.pool @bc__pool(%[[T12]]) {depth = 2 : i32
// CHECK:   aie.objectfifo.core_endpoint @bc__prod(%[[T12]]) fills @bc__pool
// CHECK:   aie.objectfifo.dma_endpoint @bc__prod__dma(%[[T12]]) drains @bc__pool
// CHECK:   aie.objectfifo.pool @bc__cons0__pool(%[[T33]]) {depth = 2 : i32
// CHECK:   aie.objectfifo.core_endpoint @bc__cons0(%[[T33]]) drains @bc__cons0__pool
// CHECK:   aie.objectfifo.dma_endpoint @bc__cons0__dma(%[[T33]]) fills @bc__cons0__pool
// CHECK:   aie.objectfifo.pool @bc__cons1__pool(%[[T13]]) {depth = 3 : i32
// CHECK:   aie.objectfifo.core_endpoint @bc__cons1(%[[T13]]) drains @bc__cons1__pool
// CHECK:   aie.objectfifo.dma_endpoint @bc__cons1__dma(%[[T13]]) fills @bc__cons1__pool
// CHECK:   aie.route from @bc__prod__dma to [@bc__cons0__dma, @bc__cons1__dma]

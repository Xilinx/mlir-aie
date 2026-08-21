// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tiles that share memory need no stream: both cores attach directly to a
// single pool, and the port on each access becomes redundant once the endpoint
// it names carries the role.

module @shared_memory {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %elem = aie.objectfifo.acquire @of0 (Produce, 1) : memref<16xi32>
      aie.objectfifo.release @of0 (Produce, 1)
      aie.end
    }

    %core13 = aie.core(%tile13) {
      %elem = aie.objectfifo.acquire @of0 (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @of0 (Consume, 1)
      aie.end
    }
  }
}

// CHECK-LABEL: @shared_memory
// CHECK-DAG:   %[[T12:.*]] = aie.tile(1, 2)
// CHECK-DAG:   %[[T13:.*]] = aie.tile(1, 3)
// CHECK:   aie.objectfifo.pool @of0_pool(%[[T12]]) {depth = 4 : i32, fifoName = "of0", segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
// CHECK:   aie.objectfifo.core_endpoint @of0_prod(%[[T12]]) fills @of0_pool
// CHECK:   aie.objectfifo.core_endpoint @of0_cons(%[[T13]]) drains @of0_pool
// CHECK-NOT: aie.objectfifo.dma_endpoint
// CHECK-NOT: aie.objectfifo.flow
// CHECK:   aie.core(%[[T12]])
// CHECK:     aie.objectfifo.acquire @of0_prod
// CHECK:     aie.objectfifo.release @of0_prod(1)
// CHECK:   aie.core(%[[T13]])
// CHECK:     aie.objectfifo.acquire @of0_cons
// CHECK:     aie.objectfifo.release @of0_cons(1)

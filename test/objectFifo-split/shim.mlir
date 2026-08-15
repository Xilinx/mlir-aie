// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A shim tile has no memory to hold objects, so its endpoint carries no pool.
// The runtime sequence drives the fifo through that endpoint.

module @shim {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of_in (%shim, {%tile02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.runtime_sequence(%arg0: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @of_in} : memref<16xi32>
      aiex.npu.dma_wait { symbol = @of_in }
    }
  }
}

// CHECK-LABEL: @shim
// CHECK-DAG:   %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK-DAG:   %[[T02:.*]] = aie.tile(0, 2)
// A shim end holds its objects in DDR, so its pool names no buffers.
// CHECK:   aie.objectfifo.pool @of_in_pool(%[[SHIM]]) {depth = 2 : i32, fifoName = "of_in", segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
// CHECK:   aie.objectfifo.dma_endpoint @of_in_prod_dma(%[[SHIM]]) drains @of_in_pool {fifoName = "of_in"}
// CHECK:   aie.objectfifo.pool @of_in_cons_pool(%[[T02]]) {depth = 2 : i32, fifoName = "of_in", segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
// CHECK:   aie.objectfifo.dma_endpoint @of_in_cons_dma(%[[T02]]) fills @of_in_cons_pool
// CHECK:   aie.objectfifo.flow from @of_in_prod_dma to [@of_in_cons_dma]
// CHECK:   aie.runtime_sequence
// CHECK:     aiex.npu.dma_memcpy_nd({{.*}}metadata = @of_in_prod_dma
// CHECK:     aiex.npu.dma_wait {symbol = @of_in_prod_dma}

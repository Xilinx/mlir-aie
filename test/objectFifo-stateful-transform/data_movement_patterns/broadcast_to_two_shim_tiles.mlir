//===- broadcast_to_two_shim_tiles.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll --split-input-file %s | FileCheck %s

// A compute tile broadcasting to two shim tiles. Each shim consumer is a
// different tile and channel, so each needs its own aie.shim_dma_allocation
// for the runtime to address.

// CHECK-LABEL: @twoShimConsumers
// Routing reaches both shim tiles.
// CHECK-DAG:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}shim_pl_tile_0_0, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}shim_pl_tile_1_0, DMA : 0)

// The first shim end keeps the fifo's own record name; the second gets one of
// its own rather than being folded into the first, which named the wrong tile.
// CHECK-DAG:     aie.shim_dma_allocation @of_shim_alloc(%{{.*}}shim_pl_tile_0_0, S2MM, 0)
// CHECK-DAG:     aie.shim_dma_allocation @of_shim_alloc_0(%{{.*}}shim_pl_tile_1_0, S2MM, 0)

module @twoShimConsumers {
 aie.device(xcve2302) {
    %tile00 = aie.tile(0, 0)
    %tile10 = aie.tile(1, 0)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of (%tile02, {%tile00, %tile10}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      %e = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
      memref.store %v, %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of (Produce, 1)
      aie.end
    }
 }
}

// -----

// A runtime sequence names the fifo, and the split points it at the fifo's
// last shim end. That end must resolve to the record for its own tile; before
// each shim end had its own record it was redirected to the first one, which
// named a different tile.

// CHECK-LABEL: @twoShimConsumersRuntime
// CHECK-DAG:     aie.shim_dma_allocation @of_shim_alloc(%{{.*}}shim_noc_tile_0_0, S2MM, 0)
// CHECK-DAG:     aie.shim_dma_allocation @of_shim_alloc_0(%{{.*}}shim_noc_tile_1_0, S2MM, 0)
// CHECK-DAG:     aiex.npu.dma_memcpy_nd{{.*}}metadata = @of_shim_alloc_0
// CHECK-DAG:     aiex.npu.dma_wait {symbol = @of_shim_alloc_0}

module @twoShimConsumersRuntime {
 aie.device(npu1) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %tile02 = aie.tile(0, 2)

    aie.objectfifo @of (%tile02, {%shim0, %shim1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      %e = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
      memref.store %v, %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of (Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%out : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%out[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {id = 0 : i64, metadata = @of} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of}
    }
 }
}

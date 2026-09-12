//===- broadcast_to_two_shim_tiles.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// A compute tile broadcasting to two shim tiles. Each shim consumer needs its
// own aie.shim_dma_allocation; a lowering that stops at the first row-0
// consumer silently drops the second.

// CHECK-LABEL: @twoShimConsumers
// Routing reaches both shim tiles.
// CHECK-DAG:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}shim_pl_tile_0_0, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}shim_pl_tile_1_0, DMA : 0)

// Known limitation: the runtime record is named after the fifo, so the second
// shim consumer gets no aie.shim_dma_allocation of its own and tile(1, 0) is
// unreachable from a runtime sequence. Emitting one record per shim consumer
// is the fix; this pins today's output so that change is deliberate.
// CHECK:         aie.shim_dma_allocation @of_shim_alloc(%{{.*}}shim_pl_tile_0_0, S2MM, 0)
// CHECK-NOT:     aie.shim_dma_allocation

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

//===- via_DMA_with_cores_test.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll --split-input-file %s | FileCheck %s

// Neighbouring tiles reach each other through shared memory, and `via_DMA`
// overrides that. Both fifos below run between the same pair, so the cores
// acquire on one fifo that stays in shared memory and one that is pushed onto
// the DMAs, and the difference has to show up in the locks the cores take.

// @of_shared stays in shared memory: its objects live only on the producer.
// CHECK-LABEL: @viaDMAWithCores
// CHECK-DAG:     aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_shared_buff_0"}
// CHECK-DAG:     aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_shared_buff_1"}
// CHECK-NOT:     of_shared_cons_buff

// @of_stream is pushed onto the DMAs, so the consumer gets its own objects.
// CHECK-DAG:     aie.buffer(%{{.*}}tile_1_2) {sym_name = "of_stream_buff_0"}
// CHECK-DAG:     aie.buffer(%{{.*}}tile_1_3) {sym_name = "of_stream_cons_buff_0"}
// CHECK-DAG:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_3, DMA : 0)

// Only the forced fifo reaches the DMAs.
// CHECK:         aie.mem(%{{.*}}tile_1_2)
// CHECK:           aie.dma_start(MM2S, 0
// CHECK:           aie.dma_bd(%{{.*}}of_stream_buff_0
// CHECK:         aie.mem(%{{.*}}tile_1_3)
// CHECK:           aie.dma_start(S2MM, 0
// CHECK:           aie.dma_bd(%{{.*}}of_stream_cons_buff_0

// CHECK-LABEL: @viaDMACrossColumn
// CHECK:         aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_2_3, DMA : 0)

module @viaDMAWithCores {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_shared (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_stream (%tile12, {%tile13}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      %a = aie.objectfifo.acquire @of_shared (Produce, 1) : memref<16xi32>
      memref.store %v, %a[%c0] : memref<16xi32>
      aie.objectfifo.release @of_shared (Produce, 1)
      %b = aie.objectfifo.acquire @of_stream (Produce, 1) : memref<16xi32>
      memref.store %v, %b[%c0] : memref<16xi32>
      aie.objectfifo.release @of_stream (Produce, 1)
      aie.end
    }

    %core13 = aie.core(%tile13) {
      %c0 = arith.constant 0 : index
      %a = aie.objectfifo.acquire @of_shared (Consume, 1) : memref<16xi32>
      %va = memref.load %a[%c0] : memref<16xi32>
      aie.objectfifo.release @of_shared (Consume, 1)
      %b = aie.objectfifo.acquire @of_stream (Consume, 1) : memref<16xi32>
      %vb = memref.load %b[%c0] : memref<16xi32>
      aie.objectfifo.release @of_stream (Consume, 1)
      aie.end
    }
 }
}

// -----

// Tiles in different columns are never neighbours, so the DMA path is the only
// one on offer and `via_DMA` asks for what would happen anyway.

module @viaDMACrossColumn {
 aie.device(xcve2302) {
    %tile02 = aie.tile(0, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of (%tile02, {%tile23}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i32
      %e = aie.objectfifo.acquire @of (Produce, 1) : memref<16xi32>
      memref.store %v, %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of (Produce, 1)
      aie.end
    }

    %core23 = aie.core(%tile23) {
      %c0 = arith.constant 0 : index
      %e = aie.objectfifo.acquire @of (Consume, 1) : memref<16xi32>
      %v = memref.load %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of (Consume, 1)
      aie.end
    }
 }
}

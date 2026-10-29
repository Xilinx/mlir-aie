//===- broadcast_join_round_trip.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// One design that runs data out and back: a shim broadcast feeds two compute
// tiles, and their results join through a single MemTile on the way to the
// shim. The MemTile carries both join slices while the broadcast it started
// from is still in flight, which is what separates this from testing either
// pattern on its own.

// CHECK-LABEL: @broadcastJoinRoundTrip
// The broadcast reaches both compute tiles from the shim.
// CHECK-DAG:     aie.flow(%{{.*}}shim_noc_tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}shim_noc_tile_0_0, DMA : 0, %{{.*}}tile_0_3, DMA : 0)

// Their results come back through the MemTile and out to the shim.
// CHECK-DAG:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}mem_tile_0_1, DMA : 0)
// CHECK-DAG:     aie.flow(%{{.*}}tile_0_3, DMA : 0, %{{.*}}mem_tile_0_1, DMA : 1)
// CHECK-DAG:     aie.flow(%{{.*}}mem_tile_0_1, DMA : 0, %{{.*}}shim_noc_tile_0_0, DMA : 0)

// The MemTile takes each result into its own slice of the joined object.
// CHECK:         aie.memtile_dma(%{{.*}}mem_tile_0_1)
// CHECK:           aie.dma_start(S2MM, 0
// CHECK:           aie.dma_bd(%{{.*}}out_mem_buff_0 : memref<64xi32> offset = 0 len = 32)
// CHECK:           aie.dma_start(S2MM, 1
// CHECK:           aie.dma_bd(%{{.*}}out_mem_buff_0 : memref<64xi32> offset = 32 len = 32)

module @broadcastJoinRoundTrip {
 aie.device(npu2) {
    %tile00 = aie.tile(0, 0)
    %tile01 = aie.tile(0, 1)
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)

    aie.objectfifo @bcast (%tile00, {%tile02, %tile03}, [2, 2, 2]) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out_a (%tile02, {%tile01}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out_b (%tile03, {%tile01}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out_mem (%tile01, {%tile00}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@out_a, @out_b] -> [@out_mem] ([0, 32][])

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %in = aie.objectfifo.acquire @bcast (Consume, 1) : memref<32xi32>
      %out = aie.objectfifo.acquire @out_a (Produce, 1) : memref<32xi32>
      %v = memref.load %in[%c0] : memref<32xi32>
      memref.store %v, %out[%c0] : memref<32xi32>
      aie.objectfifo.release @out_a (Produce, 1)
      aie.objectfifo.release @bcast (Consume, 1)
      aie.end
    }

    %core03 = aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %in = aie.objectfifo.acquire @bcast (Consume, 1) : memref<32xi32>
      %out = aie.objectfifo.acquire @out_b (Produce, 1) : memref<32xi32>
      %v = memref.load %in[%c0] : memref<32xi32>
      memref.store %v, %out[%c0] : memref<32xi32>
      aie.objectfifo.release @out_b (Produce, 1)
      aie.objectfifo.release @bcast (Consume, 1)
      aie.end
    }
 }
}

//===- packet_link_distribute.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// Packet-switched routing on the two legs of a distribute link. Switching is
// chosen per fifo, so the circuit-switched leg into the MemTile and the two
// packet-switched legs out of it coexist on one link.

// CHECK-LABEL: @packetLinkDistribute
// The leg into the MemTile stays circuit-switched.
// CHECK-DAG:     aie.flow(%{{.*}}shim_noc_tile_1_0, DMA : 0, %{{.*}}mem_tile_1_1, DMA : 0)

// Each leg out of it is packet-switched, one with an id the design pinned.
// CHECK:         aie.packet_flow(0)
// CHECK:           aie.packet_source<%{{.*}}mem_tile_1_1, DMA : 0>
// CHECK:           aie.packet_dest<%{{.*}}tile_1_2, DMA : 0>
// CHECK:         aie.packet_flow(5)
// CHECK:           aie.packet_source<%{{.*}}mem_tile_1_1, DMA : 1>
// CHECK:           aie.packet_dest<%{{.*}}tile_3_3, DMA : 0>

// The headers reach the descriptors of both distribute legs.
// CHECK:         aie.memtile_dma(%{{.*}}mem_tile_1_1)
// CHECK:           aie.dma_start(MM2S, 0
// CHECK:           aie.dma_bd_packet(0, 0)
// CHECK:           aie.dma_start(MM2S, 1
// CHECK:           aie.dma_bd_packet(0, 5)

module @packetLinkDistribute {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_in (%tile10, {%tile11}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of_a (%tile11, {%tile12}, 2 : i32) {packet} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_b (%tile11, {%tile33}, 2 : i32) {packet, packet_id = 5 : i8} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of_in] -> [@of_a, @of_b] ([] [0, 16])

    %core12 = aie.core(%tile12) {
      %e = aie.objectfifo.acquire @of_a (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @of_a (Consume, 1)
      aie.end
    }

    %core33 = aie.core(%tile33) {
      %e = aie.objectfifo.acquire @of_b (Consume, 1) : memref<16xi32>
      aie.objectfifo.release @of_b (Consume, 1)
      aie.end
    }
 }
}

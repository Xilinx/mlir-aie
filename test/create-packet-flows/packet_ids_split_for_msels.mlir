//===- packet_ids_split_for_msels.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// (4,4) DMA:1 sends id 13 to itself and (4,1) DMA:3, and id 9 to (4,1) DMA:5.
// Sent south on one port, the two ids take overlapping master sets, which ties
// both to the arbiter whose other three msels the packets into (4,4)'s DMAs
// take, one msel too many. The router branches the two ids apart at (4,4).
// Reduced from router_properties.py npu2 seed 281.

// CHECK-LABEL: aie.switchbox(%tile_4_4)
// CHECK-DAG:     %[[A0:.*]] = aie.amsel<0> (0)
// CHECK-DAG:     %[[A1:.*]] = aie.amsel<1> (0)
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 13, %[[A0]])
// CHECK-NEXT:      aie.rule(31, 9, %[[A1]])

module {
  aie.device(npu2) {
    %t_4_1 = aie.tile(4, 1)
    %t_4_4 = aie.tile(4, 4)
    %b_4_1_0 = aie.buffer(%t_4_1) {sym_name = "b_4_1_0"} : memref<256xi32>
    %b_4_1_1 = aie.buffer(%t_4_1) {sym_name = "b_4_1_1"} : memref<64xi32>
    %dma_4_1 = aie.memtile_dma(%t_4_1) {
      %d0 = aie.dma_start(S2MM, 3, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_4_1_0 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 5, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_4_1_1 : memref<64xi32> offset = 0 len = 64)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %b_4_4_2 = aie.buffer(%t_4_4) {sym_name = "b_4_4_2"} : memref<256xi32>
    %b_4_4_3 = aie.buffer(%t_4_4) {sym_name = "b_4_4_3"} : memref<32xi32>
    %dma_4_4 = aie.mem(%t_4_4) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_4_4_2 : memref<256xi32> offset = 0 len = 256)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 1, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_4_4_3 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    aie.packet_flow(0) { aie.packet_source<%t_4_1, DMA : 4> aie.packet_dest<%t_4_4, DMA : 1> }
    aie.packet_flow(5) { aie.packet_source<%t_4_1, DMA : 4> aie.packet_dest<%t_4_4, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t_4_1, DMA : 5> aie.packet_dest<%t_4_4, DMA : 0> aie.packet_dest<%t_4_4, DMA : 1> }
    aie.packet_flow(9) { aie.packet_source<%t_4_4, DMA : 1> aie.packet_dest<%t_4_1, DMA : 5> }
    aie.packet_flow(13) { aie.packet_source<%t_4_4, DMA : 1> aie.packet_dest<%t_4_1, DMA : 3> aie.packet_dest<%t_4_4, DMA : 1> }
  }
}

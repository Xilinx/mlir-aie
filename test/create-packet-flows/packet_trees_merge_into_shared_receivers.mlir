//===- packet_trees_merge_into_shared_receivers.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Both sources of flow 30 send id 30 to both its receivers, so their trees may
// merge where they reach the same receivers. They meet only at the masters into
// the receivers, (2,0) South:3 to the shim's DMA:1 and (2,1) DMA:5. Counting
// the second tree there over the channel's capacity left no legal routing.
// Reduced from router_properties.py npu2 seed 1325.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_2_0)
// CHECK:         aie.masterset(South : 3, %[[S0:[0-9]+]], %[[S1:[0-9]+]])
// CHECK:         aie.packet_rules(West : 2) {
// CHECK:           aie.rule(31, 30, %[[S1]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 30, %[[S0]])
// CHECK-LABEL: aie.switchbox(%mem_tile_2_1)
// CHECK:         aie.masterset(DMA : 5, %[[D0:[0-9]+]], %[[D1:[0-9]+]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 30, %[[D1]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 30, %[[D0]])

module {
  aie.device(npu2_3col) {
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_2_0 = aie.tile(2, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    aie.packet_flow(1) { aie.packet_source<%t_2_4, Core : 0> aie.packet_dest<%t_2_0, DMA : 1> }
    aie.packet_flow(30) { aie.packet_source<%t_1_1, DMA : 5> aie.packet_source<%t_2_5, Core : 0> aie.packet_dest<%t_2_0, DMA : 1> aie.packet_dest<%t_2_1, DMA : 5> }
    aie.packet_flow(25) { aie.packet_source<%t_1_1, DMA : 5> aie.packet_dest<%t_2_0, DMA : 1> aie.packet_dest<%t_2_4, Core : 0> }
    aie.packet_flow(0) { aie.packet_source<%t_1_0, DMA : 0> aie.packet_dest<%t_2_1, DMA : 5> }
    aie.shim_dma_allocation @in1_0(%t_1_0, MM2S, 0, <pkt_id = 0, pkt_type = 0>)
    aie.shim_dma_allocation @out1_1(%t_1_0, S2MM, 1)
    aie.shim_dma_allocation @out2_1(%t_2_0, S2MM, 1)
    aie.runtime_sequence @seq0(%a0: memref<16xi32>, %a1: memref<64xi32>, %a2: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%a0[0, 0, 0, 0][1, 1, 1, 16][16, 0, 0, 1]) { metadata = @out1_1, id = 0 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%a1[0, 0, 0, 0][1, 1, 1, 64][64, 0, 0, 1], packet = <pkt_id = 0, pkt_type = 0>) { metadata = @in1_0, id = 1 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%a2[0, 0, 0, 0][1, 1, 1, 16][16, 0, 0, 1]) { metadata = @out2_1, id = 2 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out1_1}
      aiex.npu.dma_wait {symbol = @out2_1}
    }
  }
}

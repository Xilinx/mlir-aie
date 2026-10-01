//===- priority_trees_share_channel.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Alone, the prioritized sources (0,1) DMA:0 and DMA:5 share a channel up to
// (0,2), where both meet Core:0. Flow 3 reaches Core:0 too, and was added
// between them, so packet groups made in the order flows were added put the
// two in different groups, which may not share a channel. Their pinned trees
// then overused it. Flows sharing a destination, however indirectly, are one
// group. Reduced from router_mutation.py seed 2470.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[A:.*]] = aie.amsel<{{[0-9]}}> ({{[0-9]}})
// CHECK:         aie.masterset(North : {{[0-9]}}, %[[A]]) {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 12, %[[A]]) {priority_route}
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule({{[0-9]+}}, {{[0-9]+}}, %[[A]]) {priority_route}

module {
  aie.device(npu2_3col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_1_1 = aie.tile(1, 1)
    %t_1_3 = aie.tile(1, 3)
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<16xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %d0 = aie.dma_start(MM2S, 5, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 30>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.packet_flow(3) { aie.packet_source<%t_1_1, DMA : 1> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(29) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_1_3, Core : 0> } {priority_route = true}
    aie.packet_flow(30) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(12) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> } {priority_route = true}
  }
}

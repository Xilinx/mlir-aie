//===- priority_hold_cycle_moves_the_others.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Prioritized flows 17 and 22 keep the routes they take alone, which meet at
// (1,2). Packets of 17 draining into (0,2) wait on the core there, whose
// flows the router first lays so that sharing arbiters closes a cycle through
// both prioritized flows at (1,2) as well as through flows it can move. The
// arbiter search meets the cycle through 17 and 22 first and cannot break
// it, so the router has to move the flows of every cycle it met, not just
// the first; then 17 and 22 take arbiters of their own at (1,2).

// CHECK-LABEL: aie.switchbox(%tile_1_2)
// CHECK-DAG:     %[[A4:.*]] = aie.amsel<4> (3)
// CHECK-DAG:     %[[A5:.*]] = aie.amsel<5> (3)
// CHECK:         aie.packet_rules(West : {{[0-9]}}) {
// CHECK-NEXT:      aie.rule(31, 17, %[[A4]])
// CHECK:         aie.packet_rules(North : {{[0-9]}}) {
// CHECK-NEXT:      aie.rule(31, 22, %[[A5]])

module {
  aie.device(npu1_2col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_5 = aie.tile(1, 5)
    %b_0_2_0 = aie.buffer(%t_0_2) {sym_name = "b_0_2_0"} : memref<16xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_2_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.packet_flow(22) { aie.packet_source<%t_1_5, DMA : 0> aie.packet_dest<%t_1_2, Core : 0> } {priority_route = true}
    aie.packet_flow(16) { aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_1_3, DMA : 1> }
    aie.packet_flow(24) { aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_3, Core : 0> }
    aie.packet_flow(21) { aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(12) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_1_5, DMA : 1> }
    aie.packet_flow(17) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> aie.packet_dest<%t_1_1, DMA : 5> } {priority_route = true}
  }
}

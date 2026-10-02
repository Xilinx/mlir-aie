//===- packet_tree_branches_on_its_own_id.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty

// (0,0) DMA:1 sends id 2 to (0,3) DMA:0 and, in a priority flow, on up the
// column. Its tree branches at (0,3), and a flow's own id already on a port
// counts as no second stream there. Reduced from router_mutation.py npu2
// seed 2132.
//
// (0,4) Core:0 sends id 2 to the priority flow's three destinations too, so
// the two trees meet at (0,4) before either branches for them. (0,3) DMA:0,
// which only (0,0) DMA:1 reaches, does not keep them apart.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK:         %[[BRANCH:.*]] = aie.amsel<5> (3)
// CHECK:         aie.masterset(DMA : 0, %[[BRANCH]])
// CHECK:         aie.masterset(North : [[UP:[0-9]+]], %[[BRANCH]])
// CHECK:         aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 2, %[[BRANCH]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK:         %[[MEET:.*]] = aie.amsel<5> (3)
// CHECK:         aie.masterset(DMA : 0, %[[MEET]])
// CHECK:         aie.masterset(South : {{[0-9]+}}, %[[MEET]])
// CHECK:         aie.masterset(North : 0, %[[MEET]])
// CHECK:         aie.packet_rules(Core : 0) {
// CHECK-NEXT:      aie.rule(31, 2, %[[MEET]])
// CHECK:         aie.packet_rules(South : [[UP]]) {
// CHECK-NEXT:      aie.rule(31, 2, %[[MEET]])

// NOWARN-NOT: {{warning|error}}

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %l_0_1_0 = aie.lock(%t_0_1, 0) {init = 2 : i32, sym_name = "l_0_1_0"}
    %l_0_1_1 = aie.lock(%t_0_1, 1) {init = 0 : i32, sym_name = "l_0_1_1"}
    %l_0_1_2 = aie.lock(%t_0_1, 2) {init = 2 : i32, sym_name = "l_0_1_2"}
    %l_0_1_3 = aie.lock(%t_0_1, 3) {init = 0 : i32, sym_name = "l_0_1_3"}
    %l_0_4_0 = aie.lock(%t_0_4, 0) {init = 1 : i32, sym_name = "l_0_4_0"}
    %l_0_4_1 = aie.lock(%t_0_4, 1) {init = 0 : i32, sym_name = "l_0_4_1"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<8xi32>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<16xi32>
    %b_0_1_2 = aie.buffer(%t_0_1) {sym_name = "b_0_1_2"} : memref<8xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^p1)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(S2MM, 4, ^p1b0, ^p2)
    ^p1b0:
      aie.use_lock(%l_0_1_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_1_1 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_1_1, Release, %c1)
      aie.next_bd ^p1b0
    ^p2:
      %d2 = aie.dma_start(S2MM, 5, ^p2b0, ^end)
    ^p2b0:
      aie.use_lock(%l_0_1_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_1_2 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_1_3, Release, %c1)
      aie.next_bd ^p2b0
    ^end:
      aie.end
    }
    %b_0_4_3 = aie.buffer(%t_0_4) {sym_name = "b_0_4_3"} : memref<16xi32>
    %b_0_4_4 = aie.buffer(%t_0_4) {sym_name = "b_0_4_4"} : memref<32xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_0_4_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_4_3 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_0_4_1, Release, %c1)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 1, ^p1b0, ^end)
    ^p1b0:
      aie.dma_bd(%b_0_4_4 : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 23>}
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    aie.packet_flow(23) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> }
    aie.packet_flow(9) { aie.packet_source<%t_0_4, DMA : 1> aie.packet_dest<%t_0_1, DMA : 4> }
    aie.packet_flow(29) { aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_3, Core : 0> }
    aie.packet_flow(11) { aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(27) { aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_1, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_3, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_source<%t_0_4, Core : 0> aie.packet_dest<%t_0_1, DMA : 4> aie.packet_dest<%t_0_4, DMA : 0> aie.packet_dest<%t_0_5, DMA : 1> } {priority_route = true}
  }
}


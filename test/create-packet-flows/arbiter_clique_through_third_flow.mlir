//===- arbiter_clique_through_third_flow.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=NOHOPS
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Tile (4,1) receives three packet flows and sends four to (0,1). With every
// hop packet-switched, all seven take an arbiter at (4,1), which has six.
// Flows 1 and 2 cannot deadlock each other directly: one core sends both and
// another receives both. But flow 1 can stall on (0,1), whose draining waits
// on flow 26, whose draining waits on flow 2, so the two sharing an arbiter
// closes a cycle all the same. The router says so before searching for a
// routing.

// NOHOPS: error: Unable to find a legal routing: at tile (4, 1), no two of
// NOHOPS-SAME: packet flow (4, 1) Core:0 -> (0, 1) Core:0 (id 1), packet flow (4, 1) Core:1 -> (0, 1) Core:1 (id 2),
// NOHOPS-SAME: can share an arbiter, and each takes one there whatever the routing, but the switchbox has 6 free.
// NOHOPS-SAME: For example, packet flow (4, 1) Core:0 -> (0, 1) Core:0 (id 1) can fill its receiver, and draining that waits on (0, 1) core, then (0, 1) S2MM 1, which receives packet flow (4, 1) DMA:0 -> (0, 1) DMA:1 (id 26).
// NOHOPS-SAME: packet flow (4, 1) DMA:0 -> (0, 1) DMA:1 (id 26) can fill its receiver, and draining that waits on (0, 1) core, which receives packet flow (4, 1) Core:1 -> (0, 1) Core:1 (id 2).

// With hops circuit switched, flow 1 leaves (4,1) on a circuit.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_4_1)
// CHECK:         aie.connect<Core : 0, West : 0>

module {
  aie.device(xcvc1902) {
    %t_0_1 = aie.tile(0, 1)
    %t_4_1 = aie.tile(4, 1)
    %t_20_2 = aie.tile(20, 2)
    %t_22_5 = aie.tile(22, 5)
    %t_26_6 = aie.tile(26, 6)
    %l_4_1_0 = aie.lock(%t_4_1, 0) {init = 0 : i32, sym_name = "l_4_1_0"}
    %l_4_1_1 = aie.lock(%t_4_1, 1) {init = 0 : i32, sym_name = "l_4_1_1"}
    %b_4_1_0 = aie.buffer(%t_4_1) {sym_name = "b_4_1_0"} : memref<16xi32>
    %b_4_1_1 = aie.buffer(%t_4_1) {sym_name = "b_4_1_1"} : memref<16xi32>
    %b_4_1_2 = aie.buffer(%t_4_1) {sym_name = "b_4_1_2"} : memref<8xi32>
    %b_4_1_3 = aie.buffer(%t_4_1) {sym_name = "b_4_1_3"} : memref<8xi32>
    %dma_4_1 = aie.mem(%t_4_1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^p1)
    ^p0b0:
      aie.use_lock(%l_4_1_0, Acquire, %c0)
      aie.dma_bd(%b_4_1_0 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_4_1_0, Release, %c1)
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 0, ^p1b0, ^p2)
    ^p1b0:
      aie.use_lock(%l_4_1_1, Acquire, %c1)
      aie.dma_bd(%b_4_1_1 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
      aie.use_lock(%l_4_1_1, Release, %c0)
      aie.next_bd ^end
    ^p2:
      %d2 = aie.dma_start(MM2S, 1, ^p2b0, ^end)
    ^p2b0:
      aie.dma_bd(%b_4_1_2 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 19>}
      aie.next_bd ^p2b1
    ^p2b1:
      aie.dma_bd(%b_4_1_3 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 19>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %core_4_1 = aie.core(%t_4_1) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_4_1_1, Acquire, %c0)
      aie.use_lock(%l_4_1_0, Acquire, %c1)
      aie.use_lock(%l_4_1_0, Release, %c0)
      aie.use_lock(%l_4_1_1, Release, %c1)
      aie.end
    }
    aie.packet_flow(1) { aie.packet_source<%t_4_1, Core : 0> aie.packet_dest<%t_0_1, Core : 0> } {keep_pkt_header = true}
    aie.packet_flow(2) { aie.packet_source<%t_4_1, Core : 1> aie.packet_dest<%t_0_1, Core : 1> }
    aie.packet_flow(7) { aie.packet_source<%t_22_5, Core : 0> aie.packet_dest<%t_4_1, Core : 1> }
    aie.packet_flow(8) { aie.packet_source<%t_20_2, Core : 0> aie.packet_dest<%t_4_1, Core : 0> } {keep_pkt_header = true}
    aie.packet_flow(31) { aie.packet_source<%t_26_6, DMA : 1> aie.packet_dest<%t_4_1, DMA : 1> }
    aie.packet_flow(26) { aie.packet_source<%t_4_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 1> }
    aie.packet_flow(19) { aie.packet_source<%t_4_1, DMA : 1> aie.packet_dest<%t_0_1, DMA : 0> }
  }
}

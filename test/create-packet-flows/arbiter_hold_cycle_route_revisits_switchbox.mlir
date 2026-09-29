//===- arbiter_hold_cycle_route_revisits_switchbox.mlir --------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s

// Flow 9 can reach (3,3) by passing it, turning at (3,1) and coming back up,
// which passes (3,2) and (3,3) twice: seven hops in a tree of five
// switchboxes. The router must follow every hop to see that the route then
// holds arbiter 5 at (3,3), which prioritized flow 8 needs, while its
// packets back into (3,5) wait on S2MM 0 there. Walks bounded by the number
// of switchboxes stopped short of it and emitted that hold cycle. The design
// routes if flow 8 may move. It may not, and the tree it keeps with flow 23
// already puts flow 8 on the arbiter flow 9 takes into (3,3) DMA:0, so the
// design is rejected before any routing. Reduced from router_mutation.py seed
// 430.

// CHECK: error: Unable to find a legal routing: at tile (3, 3), the routes prioritized flows (priority_route) keep put packet flow (3, 2) DMA:0 -> (3, 5) DMA:0 (id 8) and packet flow (3, 5) DMA:1 -> (3, 3) DMA:0 (id 9) on one arbiter:
// CHECK-SAME: which takes one packet rule on South:5 with packet flow (3, 2) DMA:0 -> (3, 3) DMA:0 (id 23), which leaves by DMA:0 with packet flow (3, 5) DMA:1 -> (3, 3) DMA:0 (id 9)
// CHECK-SAME: draining that waits on (3, 5) S2MM 0, which receives packet flow (3, 2) DMA:0 -> (3, 5) DMA:0 (id 8).

module {
  aie.device(npu2_4col) {
    %t_3_2 = aie.tile(3, 2)
    %t_3_3 = aie.tile(3, 3)
    %t_3_5 = aie.tile(3, 5)
    %l_3_5_0 = aie.lock(%t_3_5, 0) {init = 1 : i32, sym_name = "l_3_5_0"}
    %l_3_5_1 = aie.lock(%t_3_5, 1) {init = 0 : i32, sym_name = "l_3_5_1"}
    %b_3_5_0 = aie.buffer(%t_3_5) {sym_name = "b_3_5_0"} : memref<24xi32>
    %dma_3_5 = aie.mem(%t_3_5) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_3_5_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_3_5_0 : memref<24xi32> offset = 0 len = 24)
      aie.use_lock(%l_3_5_1, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.packet_flow(23) { aie.packet_source<%t_3_2, DMA : 0> aie.packet_dest<%t_3_3, DMA : 0> aie.packet_dest<%t_3_5, DMA : 0> }
    aie.packet_flow(8) { aie.packet_source<%t_3_2, DMA : 0> aie.packet_dest<%t_3_5, DMA : 0> } {priority_route = true}
    aie.packet_flow(9) { aie.packet_source<%t_3_5, DMA : 1> aie.packet_dest<%t_3_3, DMA : 0> aie.packet_dest<%t_3_5, DMA : 1> }
  }
}

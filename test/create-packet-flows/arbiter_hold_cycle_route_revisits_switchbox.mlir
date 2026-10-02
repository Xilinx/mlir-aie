//===- arbiter_hold_cycle_route_revisits_switchbox.mlir --------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN --allow-empty

// Prioritized flow 23 branches at (3,3), one branch into its DMA:0 and one
// going on North. Flow 9 routes only if flow 23 reaches DMA:0 by a master set
// of its own, which it does not alone, so without a control-packet reload,
// flow 23 routes like the others. It reaches DMA:0 by passing (3,3), turning
// at (2,3) and coming back, so its tree enters (3,3) twice, which hold-cycle
// walks have to follow to the end. On one arbiter there, the branch holding it
// waits on the other, as both move as one, so they take two. Flow 9 shares the
// arbiter into (3,3) DMA:0 with flow 23, which deadlocks only if the copy it
// sends itself on (3,5) S2MM 1, which nothing drains, overruns it.
// Reduced from router_mutation.py seed 430.

// CHECK-LABEL: aie.switchbox(%tile_3_3)
// CHECK-DAG:     %[[INTO:.*]] = aie.amsel<1> (0)
// CHECK-DAG:     %[[OUT:.*]] = aie.amsel<0> (1)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[INTO]])
// CHECK-DAG:     aie.masterset(West : 1, %[[OUT]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 9, %[[INTO]])
// CHECK:         aie.packet_rules(South : 4) {
// CHECK:           aie.rule(31, 23, %[[OUT]])
// CHECK:         aie.packet_rules(West : 1) {
// CHECK-NEXT:      aie.rule(31, 23, %[[INTO]])

// WARN-NOT: {{warning|error}}

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
    aie.packet_flow(23) { aie.packet_source<%t_3_2, DMA : 0> aie.packet_dest<%t_3_3, DMA : 0> aie.packet_dest<%t_3_5, DMA : 0> } {priority_route = true}
    aie.packet_flow(8) { aie.packet_source<%t_3_2, DMA : 0> aie.packet_dest<%t_3_5, DMA : 0> } {priority_route = true}
    aie.packet_flow(9) { aie.packet_source<%t_3_5, DMA : 1> aie.packet_dest<%t_3_3, DMA : 0> aie.packet_dest<%t_3_5, DMA : 1> }
  }
}

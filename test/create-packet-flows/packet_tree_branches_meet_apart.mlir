//===- packet_tree_branches_meet_apart.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN --allow-empty

// Flow 28's packets from (0,5) go down to (0,1), back up, and join the packets
// from (0,3) at (0,3), so their tree comes into (0,2) twice, from North:3 and
// North:1. A tree's branches move as one: on one arbiter there, a packet
// holding it on its way down waits on itself on its way back, and the design
// hangs on hardware. So each takes its own. Reduced from router_properties.py
// npu2 seed 232.

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK-DAG:     %[[DOWN:.*]] = aie.amsel<3> (3)
// CHECK-DAG:     %[[BACK:.*]] = aie.amsel<4> (3)
// CHECK-DAG:     %[[UP:.*]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(South : 0, %[[BACK]])
// CHECK-DAG:     aie.masterset(South : 2, %[[DOWN]])
// CHECK-DAG:     aie.masterset(North : 1, %[[UP]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 28, %[[DOWN]])
// CHECK:         aie.packet_rules(South : 2) {
// CHECK-NEXT:      aie.rule(31, 28, %[[UP]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 28, %[[BACK]])

// WARN-NOT: {{warning|error}}

module {
  aie.device(npu2_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.packet_flow(28) { aie.packet_source<%t_0_3, Core : 0> aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_1, DMA : 4> aie.packet_dest<%t_0_4, Core : 0> } {priority_route = true}
  }
}

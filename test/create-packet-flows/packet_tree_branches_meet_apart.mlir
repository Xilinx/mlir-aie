//===- packet_tree_branches_meet_apart.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// Prioritized flow 27 reaches (1,2) DMA:1 by a port of its own, so its tree
// branches at (1,0) and both branches go up through (1,1) and (1,2). The
// branches move as one: on one arbiter there, the one holding it waits on the
// other, and the design hangs on hardware. So each takes its own. Flow 9
// routes only if flow 27 reaches DMA:1 by a master set of its own, which it
// does not alone, so a reload would not keep it. Reduced from
// router_properties.py npu2 seed 276.

// CHECK-LABEL: aie.switchbox(%tile_1_2)
// CHECK-DAG:     %[[ON2:.*]] = aie.amsel<4> (3)
// CHECK-DAG:     %[[INTO:.*]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(DMA : 1, %[[INTO]])
// CHECK-DAG:     aie.masterset(North : 4, %[[ON2]])
// CHECK:         aie.packet_rules(West : 1) {
// CHECK-NEXT:      aie.rule(31, 9, %[[INTO]])
// CHECK:         aie.packet_rules(South : 5) {
// CHECK-NEXT:      aie.rule(31, 27, %[[ON2]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 27, %[[INTO]])
// CHECK-LABEL: aie.switchbox(%mem_tile_1_1)
// CHECK-DAG:     %[[ON1:.*]] = aie.amsel<4> (3)
// CHECK-DAG:     %[[UP:.*]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(North : 1, %[[UP]])
// CHECK-DAG:     aie.masterset(North : 5, %[[ON1]])
// CHECK:         aie.packet_rules(South : 5) {
// CHECK-NEXT:      aie.rule(31, 27, %[[ON1]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 27, %[[UP]])

// WARN: warning: the prioritized flows (the control overlay) take another route than they take alone
// WARN-NOT: {{warning|error}}

module {
  aie.device(npu2) {
    %t_0_1 = aie.tile(0, 1)
    %t_1_0 = aie.tile(1, 0)
    %t_1_2 = aie.tile(1, 2)
    %t_1_4 = aie.tile(1, 4)
    aie.packet_flow(27) { aie.packet_source<%t_1_0, DMA : 1> aie.packet_dest<%t_1_2, DMA : 1> aie.packet_dest<%t_1_4, DMA : 1> } {priority_route = true}
    aie.packet_flow(9) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_1_2, DMA : 1> }
  }
}

//===- packet_tree_branches_off_shared_arbiter.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// (1,2) DMA:1 sends id 2 to (0,3) and id 30 to (0,1) and back to itself,
// where id 11 from (1,5) arrives too. Ids 2 and 11 can deadlock on a shared
// arbiter, and ids 30 and 11 share the one of DMA:1. Ids 2 and 30 both head
// West, and on one master port they would share its arbiter, so the tree
// branches at (1,2) and each id takes its own. Reduced from
// router_mutation.py seed 2377.

// CHECK-LABEL: aie.switchbox(%tile_1_2)
// CHECK:         %[[A:.*]] = aie.amsel<0> (0)
// CHECK:         %[[B:.*]] = aie.amsel<1> (0)
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 30, %[[A]])
// CHECK-NEXT:      aie.rule(31, 2, %[[B]])

module {
  aie.device(npu2_3col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_3 = aie.tile(0, 3)
    %t_1_2 = aie.tile(1, 2)
    %t_1_5 = aie.tile(1, 5)
    aie.packet_flow(2) { aie.packet_source<%t_1_2, DMA : 1> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(30) { aie.packet_source<%t_1_2, DMA : 1> aie.packet_dest<%t_0_1, DMA : 0> aie.packet_dest<%t_1_2, DMA : 1> }
    aie.packet_flow(11) { aie.packet_source<%t_1_5, Core : 0> aie.packet_dest<%t_1_2, DMA : 1> }
  }
}

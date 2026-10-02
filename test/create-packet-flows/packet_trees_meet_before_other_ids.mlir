//===- packet_trees_meet_before_other_ids.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty

// (0,4) and (1,4) send id 14 to (0,3) Core:1 and (0,8) DMA:1, where (1,2)
// sends ids 29 and 1. The two id 14 trees have to meet on the way there, so
// neither meets the tree of (1,2) for those destinations, and they meet at
// (0,4). Reduced from router_properties.py xcvc1902 seed 15.

// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(South : {{[0-9]+}}, %[[MEET]])
// CHECK:         aie.masterset(North : {{[0-9]+}}, %[[MEET]])
// CHECK:         aie.packet_rules(Core : 1) {
// CHECK-NEXT:      aie.rule(31, 14, %[[MEET]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(East : 0) {
// CHECK-NEXT:      aie.rule(31, 14, %[[MEET]])
// CHECK-NEXT:    }

// NOWARN-NOT: {{warning|error}}

module {
  aie.device(xcvc1902) {
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_8 = aie.tile(0, 8)
    %t_1_2 = aie.tile(1, 2)
    %t_1_4 = aie.tile(1, 4)
    aie.packet_flow(14) { aie.packet_source<%t_1_4, Core : 1> aie.packet_source<%t_0_4, Core : 1> aie.packet_dest<%t_0_3, Core : 1> aie.packet_dest<%t_0_8, DMA : 1> }
    aie.packet_flow(29) { aie.packet_source<%t_1_2, DMA : 0> aie.packet_dest<%t_0_3, Core : 1> }
    aie.packet_flow(1) { aie.packet_source<%t_1_2, DMA : 0> aie.packet_dest<%t_0_8, DMA : 1> }
  }
}

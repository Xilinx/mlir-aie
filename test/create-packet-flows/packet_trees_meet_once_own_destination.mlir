//===- packet_trees_meet_once_own_destination.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty

// (2,4) DMA:0 and (2,5) DMA:0 send id 3 to (2,2) and to (2,5) DMA:0 itself.
// The tree from (2,5) meets the one from (2,4) at (2,4) alone, by way of
// (1,4), and its packets return to (2,5) on the tree they meet there.
// Reduced from router_properties.py npu2 seed 197.

// CHECK-LABEL: aie.switchbox(%tile_2_4)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (0)
// CHECK:         %[[AROUND:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(South : 0, %[[MEET]])
// CHECK:         aie.masterset(West : 0, %[[AROUND]])
// CHECK:         aie.masterset(North : 5, %[[MEET]])
// CHECK:         aie.packet_rules(West : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[MEET]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[MEET]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[AROUND]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_2_5)
// CHECK:         %[[OUT:.*]] = aie.amsel<0> (0)
// CHECK:         %[[BACK:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(DMA : 0, %[[BACK]])
// CHECK:         aie.masterset(South : 0, %[[OUT]])
// CHECK:         aie.packet_rules(South : 5) {
// CHECK-NEXT:      aie.rule(31, 3, %[[BACK]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[OUT]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_1_4)
// CHECK:         aie.packet_rules(East : 0) {
// CHECK-NEXT:      aie.rule(31, 3,
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// NOWARN-NOT: {{warning|error}}

module {
  aie.device(npu2) {
    %t_2_2 = aie.tile(2, 2)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    aie.packet_flow(3) { aie.packet_source<%t_2_4, DMA : 0> aie.packet_source<%t_2_5, DMA : 0> aie.packet_dest<%t_2_2, DMA : 0> aie.packet_dest<%t_2_5, DMA : 0> }
  }
}

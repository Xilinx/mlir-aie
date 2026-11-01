//===- packet_trees_meet_once.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty

// (2,1) and (2,5) send id 15 to (1,0) and (1,4). Trees meeting at two
// switchboxes can each hold an arbiter at one and wait at the other, so they
// meet at (2,4) alone and reach both destinations as one tree from there.
// Reduced from router_properties.py npu2 seed 172.

// CHECK-LABEL: aie.switchbox(%tile_1_4)
// CHECK:         %[[OUT:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 0, %[[OUT]])
// CHECK:         aie.masterset(South : 2, %[[OUT]])
// CHECK:         aie.packet_rules(East : 1) {
// CHECK-NEXT:      aie.rule(31, 15, %[[OUT]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-LABEL: aie.switchbox(%tile_2_4)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(West : 1, %[[MEET]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 15, %[[MEET]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(South : 0) {
// CHECK-NEXT:      aie.rule(31, 15, %[[MEET]])
// CHECK-NEXT:    }

// NOWARN-NOT: {{warning|error}}

module {
  aie.device(npu2) {
    %t_1_0 = aie.tile(1, 0)
    %t_1_4 = aie.tile(1, 4)
    %t_2_1 = aie.tile(2, 1)
    %t_2_5 = aie.tile(2, 5)
    aie.packet_flow(15) { aie.packet_source<%t_2_1, DMA : 1> aie.packet_source<%t_2_5, DMA : 1> aie.packet_dest<%t_1_0, DMA : 0> aie.packet_dest<%t_1_4, DMA : 0> }
  }
}

//===- packet_trees_other_ids_meet_once.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty

// (3,5) sends id 20 and (2,1) id 11 to (3,0) DMA:0 and (3,4) DMA:0. Meeting
// at both receivers, each tree could hold one and wait at the other, so with
// their other ids they still meet once, at (3,4), and go on to (3,0) as one.
// Reduced from router_properties.py npu2 seed 39.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_3_0)
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-LABEL: aie.switchbox(%tile_3_4)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 0, %[[MEET]])
// CHECK:         aie.masterset(South : 0, %[[MEET]])
// CHECK:         aie.packet_rules(South : 0) {
// CHECK-NEXT:      aie.rule(31, 11, %[[MEET]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 20, %[[MEET]])
// CHECK-NEXT:    }

// NOWARN-NOT: {{warning|error}}

module {
  aie.device(npu2) {
    %t_2_1 = aie.tile(2, 1)
    %t_3_0 = aie.tile(3, 0)
    %t_3_4 = aie.tile(3, 4)
    %t_3_5 = aie.tile(3, 5)
    aie.packet_flow(20) { aie.packet_source<%t_3_5, DMA : 1> aie.packet_dest<%t_3_0, DMA : 0> aie.packet_dest<%t_3_4, DMA : 0> }
    aie.packet_flow(11) { aie.packet_source<%t_2_1, DMA : 4> aie.packet_dest<%t_3_0, DMA : 0> aie.packet_dest<%t_3_4, DMA : 0> }
  }
}

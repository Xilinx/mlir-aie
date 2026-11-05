//===- packet_meeting_takes_own_loopback.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Ids 30 and 31 both reach (0, 2) Core:0 and (0, 1) DMA:3, which id 30 is
// sent from. Id 30 goes up to (0, 2) and meets the tree of id 31 there, by a
// U-turn on South:3, and comes back down into its own source port on it. A
// loopback inside (0, 1) would put DMA:3 and North:3 on one arbiter with id
// 28, and id 31 on (0, 2)'s Core:0 arbiter with id 28: a hold cycle.
// Reduced from router_properties.py npu1 seed 207470648.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[UP:.*]] = aie.amsel<0> (0)
// CHECK:         %[[DOWN:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(DMA : 3, %[[DOWN]])
// CHECK:         aie.masterset(North : 3, %[[UP]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(30, 30, %[[DOWN]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(DMA : 3) {
// CHECK-NEXT:      aie.rule(29, 28, %[[UP]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[CORE:.*]] = aie.amsel<0> (0)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (1)
// CHECK:         aie.masterset(Core : 0, %[[CORE]], %[[MEET]])
// CHECK:         aie.masterset(South : 3, %[[MEET]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 31, %[[MEET]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(31, 30, %[[MEET]])
// CHECK-NEXT:      aie.rule(31, 28, %[[CORE]])
// CHECK-NEXT:    }

module {
  aie.device(npu1_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    aie.packet_flow(28) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(30) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_1, DMA : 3> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(31) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_1, DMA : 3> aie.packet_dest<%t_0_2, Core : 0> }
  }
}

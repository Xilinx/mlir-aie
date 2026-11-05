//===- packet_link_waits_ordered_above.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Id 20's two sources at (2, 4) merge there and meet id 16's tree at (2, 3),
// whose arbiter 0 grants every packet of both ids the links below it in one
// order: into (2, 2), (3, 3) and (3, 4). A packet queued behind another on
// one of those links was granted that arbiter after it, so packets queued
// each behind the next on them make no cycle, though id 20's two sources
// merged at (2, 4) and not at (2, 3).
// Reduced from router_properties.py npu1 seed 207471827.

// CHECK-LABEL: aie.switchbox(%tile_2_3)
// CHECK:         %[[SIXTEEN:.*]] = aie.amsel<0> (0)
// CHECK:         %[[TWENTY:.*]] = aie.amsel<0> (1)
// CHECK:         aie.masterset(Core : 0, %[[SIXTEEN]], %[[TWENTY]])
// CHECK:         aie.masterset(South : 3, %[[TWENTY]])
// CHECK:         aie.masterset(East : 0, %[[SIXTEEN]], %[[TWENTY]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 20, %[[TWENTY]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(West : 0) {
// CHECK-NEXT:      aie.rule(31, 16, %[[SIXTEEN]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(31, 16, %[[SIXTEEN]])
// CHECK-NEXT:    }

module {
  aie.device(npu1) {
    %t_1_0 = aie.tile(1, 0)
    %t_1_3 = aie.tile(1, 3)
    %t_2_2 = aie.tile(2, 2)
    %t_2_3 = aie.tile(2, 3)
    %t_2_4 = aie.tile(2, 4)
    %t_3_4 = aie.tile(3, 4)
    aie.packet_flow(16) { aie.packet_source<%t_1_0, DMA : 0> aie.packet_source<%t_1_3, DMA : 1> aie.packet_dest<%t_2_3, Core : 0> aie.packet_dest<%t_3_4, DMA : 0> }
    aie.packet_flow(20) { aie.packet_source<%t_2_4, DMA : 1> aie.packet_source<%t_2_4, Core : 0> aie.packet_dest<%t_2_2, DMA : 1> aie.packet_dest<%t_2_3, Core : 0> aie.packet_dest<%t_3_4, DMA : 0> }
  }
}

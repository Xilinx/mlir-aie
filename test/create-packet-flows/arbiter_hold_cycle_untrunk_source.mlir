//===- arbiter_hold_cycle_untrunk_source.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN --allow-empty

// Flows 31 and 6 both start at (1,2) and share one destination there, DMA:1.
// Reaching it by one master port out of (1,2), a tree would come back into
// (1,2) and wait there on the arbiter its own packet holds. Each tree branches
// at (1,2) instead, so nothing re-enters it.
// Reduced from router_mutation.py npu1 seed 203661246.

// CHECK-LABEL: aie.switchbox(%tile_1_2)
// CHECK-DAG:     %[[A:.*]] = aie.amsel<0> (0)
// CHECK-DAG:     %[[B:.*]] = aie.amsel<0> (1)
// CHECK-DAG:     aie.masterset(DMA : 1, %[[A]], %[[B]])
// CHECK-DAG:     aie.masterset(East : {{[0-9]+}}, %[[A]])
// CHECK-DAG:     aie.masterset(East : {{[0-9]+}}, %[[B]])
// CHECK-NOT:     aie.packet_rules(East
// CHECK:         aie.switchbox(%mem_tile_2_1)

// WARN-NOT: {{warning|error}}

module {
  aie.device(npu1_3col) {
    %t_1_2 = aie.tile(1, 2)
    %t_2_1 = aie.tile(2, 1)
    aie.packet_flow(31) {
      aie.packet_source<%t_1_2, Core : 0>
      aie.packet_dest<%t_1_2, DMA : 1>
      aie.packet_dest<%t_2_1, DMA : 4>
    }
    aie.packet_flow(6) {
      aie.packet_source<%t_1_2, DMA : 1>
      aie.packet_dest<%t_1_2, DMA : 1>
      aie.packet_dest<%t_2_1, DMA : 2>
    }
  }
}

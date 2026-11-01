//===- packet_tree_stays_whole_beside_a_hazard.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Id 18 from (0,5) must not share an arbiter at (0,3) with id 16 from (0,3)
// DMA:1. Branching (0,3)'s tree to put its ids 16 and 18 apart does not help,
// as id 16 meets id 18 from (0,5) on its own, and it would need five channels
// down from (0,2), which has four. Id 18 from (0,5) takes its own channel
// instead. Reduced from router_properties.py npu2 seed 360.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK:           aie.rule(31, 18, {{%[0-9]+}})
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(29, 16, {{%[0-9]+}})
// CHECK-NEXT:    }

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.packet_flow(31) { aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_0_0, DMA : 1> }
    aie.packet_flow(16) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_0, DMA : 0> }
    aie.packet_flow(18) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_1, DMA : 2> }
    aie.packet_flow(1) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_0_1, DMA : 1> }
  }
}

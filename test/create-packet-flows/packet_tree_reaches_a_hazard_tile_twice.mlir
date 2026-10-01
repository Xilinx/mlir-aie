//===- packet_tree_reaches_a_hazard_tile_twice.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Id 23 from (0,1) DMA:3 loops back to (0,1) DMA:3 and must not share an
// arbiter with id 9, which (0,0) DMA:0 sends up the column alongside id 23.
// Id 23 for (0,1) DMA:3 reaches (0,1) by a slave port of its own, so id 9
// stays off the DMA:3 arbiter while ids 9 and 23 still share the rest of
// the tree. Reduced from router_mutation.py npu2 seed 1128.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.masterset(DMA : 3, %[[LOOP:[0-9]+]], %[[SHIM:[0-9]+]])
// CHECK:         aie.packet_rules(DMA : 3) {
// CHECK-NEXT:      aie.rule(31, 23, %[[LOOP]])
// CHECK-NEXT:    }
// CHECK:         aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 23, %[[SHIM]])
// CHECK-NEXT:    }

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_1, DMA : 0, %t_0_5, DMA : 1)
    aie.packet_flow(24) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_0_5, Core : 0> }
    aie.packet_flow(26) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_3, DMA : 0> }
    aie.packet_flow(23) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_1, DMA : 3> aie.packet_dest<%t_0_5, DMA : 0> }
    aie.packet_flow(9) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_4, DMA : 1> }
    aie.packet_flow(15) { aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 1> aie.packet_dest<%t_0_4, DMA : 0> }
  }
}

//===- arbiter_hold_cycle_splits_shared_source.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN --allow-empty

// (0,3) DMA:0 sends ids 9 and 29, and (0,1) DMA:2 sends id 9 too. Flow 29 can
// hold an arbiter flow 9 from (0,1) needs, in a cycle of waits through its
// receiver at (0,5) and those at (0,0), so the two sources route apart. Trees
// with an id in common join where they meet, so id 29 routes apart from (0,1)
// as a part of its own.
// Reduced from router_properties.py npu2 seed 4736.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK-DAG:     %[[UP:.*]] = aie.amsel<2> (0)
// CHECK-DAG:     aie.masterset(North : 4, %[[UP]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 29, %[[UP]])
// CHECK-NEXT:      aie.rule(31, 9,

// WARN-NOT: {{warning|error}}

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_5, Core : 0, %t_0_3, DMA : 1)
    aie.flow(%t_0_5, Core : 0, %t_0_1, DMA : 0)
    aie.packet_flow(9) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_source<%t_0_1, DMA : 2> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(29) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(27) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_0, DMA : 1> }
    aie.packet_flow(20) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_1, DMA : 4> }
  }
}

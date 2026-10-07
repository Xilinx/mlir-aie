//===- packet_trees_trunk_to_meet.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty

// (5,1) DMA:1 and (6,1) DMA:0 send id 27 to (5,2) and (6,0). A tree branching
// North and South at its memtile leaves no slave port there to meet it by, so
// each leaves its memtile by one master port for both, and the two meet at
// (6,0) before branching. Reduced from router_properties.py npu2 seed 220.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_5_0)
// CHECK:         %[[UP:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(North : 5, %[[UP]])
// CHECK-LABEL: aie.switchbox(%mem_tile_5_1)
// CHECK:         %[[DOWN:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(South : 1, %[[DOWN]])
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 27, %[[DOWN]])
// CHECK-LABEL: aie.switchbox(%shim_noc_tile_6_0)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(South : 2, %[[MEET]])
// CHECK:         aie.masterset(West : 3, %[[MEET]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 27, %[[MEET]])
// CHECK:         aie.packet_rules(West : 3) {
// CHECK-NEXT:      aie.rule(31, 27, %[[MEET]])
// CHECK-LABEL: aie.switchbox(%mem_tile_6_1)
// CHECK:         %[[OUT:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(South : 1, %[[OUT]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 27, %[[OUT]])

// NOWARN-NOT: {{warning|error}}

module {
  aie.device(npu2) {
    %t_5_0 = aie.tile(5, 0)
    %t_5_1 = aie.tile(5, 1)
    %t_5_2 = aie.tile(5, 2)
    %t_6_0 = aie.tile(6, 0)
    %t_6_1 = aie.tile(6, 1)
    aie.packet_flow(27) { aie.packet_source<%t_5_1, DMA : 1> aie.packet_source<%t_6_1, DMA : 0> aie.packet_dest<%t_5_2, DMA : 0> aie.packet_dest<%t_5_2, DMA : 1> aie.packet_dest<%t_6_0, DMA : 0> }
  }
}

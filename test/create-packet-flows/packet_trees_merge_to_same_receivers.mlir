//===- packet_trees_merge_to_same_receivers.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Both sources of flow 25 send id 25 to (0,3) DMA:0 and (0,2) Core:0. The tree
// from (0,5) Core:0 comes down to (0,1) and turns back north into the tree from
// (0,0) DMA:0. Both take the id to the same receivers, so they may share the
// channel. Counting the second tree over its capacity left only routings that
// can deadlock. Reduced from router_mutation.py npu2 seed 96 (permute).

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[M:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(North : {{[0-9]+}}, %[[M]])
// CHECK:         aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 25, %[[M]])
// CHECK:         aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 25, %[[M]])

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.packet_flow(25) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_3, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(27) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(21) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_0_3, DMA : 0> }
  }
}

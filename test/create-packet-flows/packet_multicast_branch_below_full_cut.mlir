//===- packet_multicast_branch_below_full_cut.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Sixteen trees go down from row 2 to row 1, which has sixteen channels, so
// packet id 19 crosses once and branches below, to (2,1) and (3,1). Mem tiles
// have no East or West ports, so the branch goes by the shim row. A branch off
// the tree used to pay again for the congestion of the hops it shares, which
// made crossing a second channel look as cheap. Reduced from
// router_mutation.py seed 490.

// Id 19 reaches (2,1) from the shim below, not across the cut at (2,2).
// CHECK-LABEL: %switchbox_2_1 = aie.switchbox(%mem_tile_2_1)
// CHECK:         %[[A21:.*]] = aie.amsel<{{[0-9]+}}> ({{[0-9]+}})
// CHECK-NEXT:    aie.masterset(DMA : 3, %[[A21]])
// CHECK-NEXT:    aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 19, %[[A21]])
// CHECK-LABEL: %switchbox_2_2 = aie.switchbox(%tile_2_2)
// CHECK-NOT:     aie.rule(31, 19,
// It crosses the cut once, at (3,2).
// CHECK-LABEL: %switchbox_3_2 = aie.switchbox(%tile_3_2)
// CHECK:         %[[A32:.*]] = aie.amsel<{{[0-9]+}}> ({{[0-9]+}})
// CHECK-NEXT:    aie.masterset(South : {{[0-9]+}}, %[[A32]])
// CHECK-NEXT:    aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 19, %[[A32]])
// CHECK-LABEL: %switchbox_2_0 = aie.switchbox(%shim_noc_tile_2_0)
// CHECK:         %[[A20:.*]] = aie.amsel<{{[0-9]+}}> ({{[0-9]+}})
// CHECK-NEXT:    aie.masterset(North : {{[0-9]+}}, %[[A20]])
// CHECK-NEXT:    aie.packet_rules(East : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 19, %[[A20]])
// CHECK-LABEL: %switchbox_3_0 = aie.switchbox(%shim_noc_tile_3_0)
// CHECK:         %[[A30:.*]] = aie.amsel<{{[0-9]+}}> ({{[0-9]+}})
// CHECK-NEXT:    aie.masterset(West : {{[0-9]+}}, %[[A30]])
// CHECK-NEXT:    aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 19, %[[A30]])
// Below the cut it branches at (3,1): to its DMA and down to the shim.
// CHECK-LABEL: %switchbox_3_1 = aie.switchbox(%mem_tile_3_1)
// CHECK:         %[[A31:.*]] = aie.amsel<{{[0-9]+}}> ({{[0-9]+}})
// CHECK-NEXT:    aie.masterset(DMA : 1, %[[A31]])
// CHECK-NEXT:    aie.masterset(South : {{[0-9]+}}, %[[A31]])
// CHECK-NEXT:    aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:      aie.rule(31, 19, %[[A31]])

module {
  aie.device(npu2_4col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_3 = aie.tile(1, 3)
    %t_2_0 = aie.tile(2, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_3 = aie.tile(2, 3)
    %t_2_5 = aie.tile(2, 5)
    %t_3_0 = aie.tile(3, 0)
    %t_3_1 = aie.tile(3, 1)
    %t_3_2 = aie.tile(3, 2)
    %t_3_3 = aie.tile(3, 3)
    %t_3_4 = aie.tile(3, 4)
    %t_3_5 = aie.tile(3, 5)
    aie.flow(%t_3_2, DMA : 0, %t_2_1, DMA : 5)
    aie.flow(%t_3_5, DMA : 1, %t_1_1, DMA : 5)
    aie.flow(%t_0_2, DMA : 0, %t_3_1, DMA : 3)
    aie.flow(%t_2_3, Core : 0, %t_2_1, DMA : 4)
    aie.flow(%t_0_3, DMA : 1, %t_1_1, DMA : 2)
    aie.flow(%t_0_5, DMA : 0, %t_1_0, DMA : 0)
    aie.flow(%t_2_5, DMA : 1, %t_0_1, DMA : 0)
    aie.flow(%t_3_2, DMA : 1, %t_3_0, DMA : 0)
    aie.flow(%t_3_3, Core : 0, %t_2_0, DMA : 0)
    aie.flow(%t_1_3, Core : 0, %t_3_1, DMA : 0)
    aie.flow(%t_0_5, DMA : 1, %t_2_1, DMA : 2)
    aie.flow(%t_2_2, DMA : 0, %t_3_0, DMA : 1)
    aie.flow(%t_2_3, DMA : 1, %t_1_0, DMA : 1)
    aie.flow(%t_1_3, DMA : 1, %t_3_1, DMA : 4)
    aie.flow(%t_2_5, DMA : 0, %t_1_1, DMA : 1)
    aie.packet_flow(19) {
      aie.packet_source<%t_3_4, DMA : 1>
      aie.packet_dest<%t_2_1, DMA : 3>
      aie.packet_dest<%t_3_1, DMA : 1>
    }
  }
}

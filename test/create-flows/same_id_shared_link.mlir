//===- same_id_shared_link.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Regression test for the pathfinder "merge-then-fanout" packet-routing bug.
//
// Two DISTINCT packet flows share the same packet id (0) and the same
// destination (mem_tile_1_1 DMA:3). The old router merged them onto a single
// amsel and then fanned that merged stream out to two switchbox output ports,
// so the two id-0 streams were delivered duplicated/indistinguishably and
// deadlocked on hardware. The bug is order-dependent: it only surfaces when the
// unrelated congestion flow (packet_flow(2), mem_tile_1_1 DMA:5 -> tile_3_5) is
// routed before the two colliding id-0 flows.
//
// With id-gated channel sharing, the two id-0 flows must take disjoint physical
// paths wherever they would otherwise share a link. The checks below pin that
// at the merge tile (tile_1_2) each id-0 stream gets its own amsel driving a
// single output port -- no amsel fans out to more than one masterset port.

// CHECK-LABEL: aie.device(npu2)

// Destination tile: both id-0 streams arrive on separate input ports (North : 2
// and North : 3) and fan in to the single DMA : 3 endpoint.
// CHECK:      %[[mem_tile_1_1:.*]] = aie.tile(1, 1)
// CHECK:      aie.switchbox(%[[mem_tile_1_1]]) {
// CHECK-NEXT:   %[[b0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[b1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   aie.masterset(DMA : 3, %[[b1]])
// CHECK-NEXT:   aie.masterset(North : 1, %[[b0]])
// CHECK-NEXT:   aie.packet_rules(North : 2) {
// CHECK-NEXT:     aie.rule(31, 0, %[[b1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(North : 3) {
// CHECK-NEXT:     aie.rule(31, 0, %[[b1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 5) {
// CHECK-NEXT:     aie.rule(31, 2, %[[b0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Merge tile: the two id-0 streams stay on distinct amsels, each driving exactly
// one output port (North : 2 -> South : 2 via %[[a1]]; East : 2 -> South : 3 via
// %[[a2]]). No single amsel fans out to two ports.
// CHECK:      %[[tile_1_2:.*]] = aie.tile(1, 2)
// CHECK:      aie.switchbox(%[[tile_1_2]]) {
// CHECK-NEXT:   %[[a0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[a1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   %[[a2:.*]] = aie.amsel<2> (0)
// CHECK-NEXT:   aie.masterset(South : 2, %[[a1]])
// CHECK-NEXT:   aie.masterset(South : 3, %[[a2]])
// CHECK-NEXT:   aie.masterset(East : 2, %[[a0]])
// CHECK-NEXT:   aie.packet_rules(North : 2) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(East : 2) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a2]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(South : 1) {
// CHECK-NEXT:     aie.rule(31, 2, %[[a0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
  aie.device(npu2) {
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    // Congestion flow -- must be routed FIRST to trigger the bug:
    aie.packet_flow(2) {
      aie.packet_source<%mem_tile_1_1, DMA : 5>
      aie.packet_dest<%tile_3_5, DMA : 0>
    }
    // Two distinct flows sharing id 0 and dest mem_tile_1_1 DMA:3 -- the router merges
    // these and fans the merged stream out to two ports:
    aie.packet_flow(0) {
      aie.packet_source<%tile_2_3, DMA : 0>
      aie.packet_dest<%mem_tile_1_1, DMA : 3>
    }
    aie.packet_flow(0) {
      aie.packet_source<%tile_2_5, DMA : 0>
      aie.packet_dest<%mem_tile_1_1, DMA : 3>
    }
  }
}

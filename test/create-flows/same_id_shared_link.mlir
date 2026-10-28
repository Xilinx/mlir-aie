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
// so the two id-0 streams were delivered duplicated and deadlocked on
// hardware. The bug is order-dependent: it only surfaces when the unrelated
// congestion flow (packet_flow(2), mem_tile_1_1 DMA:5 -> tile_3_5) is routed
// before the two colliding id-0 flows.
//
// Same-id flows to the same destination may merge, as long as the merged
// stream then takes a single path. Here tile_2_5's stream comes down into
// tile_2_3, both id-0 streams share one amsel driving one South port, and the
// merged stream reaches mem_tile_1_1 on a single North port.

// CHECK-LABEL: aie.device(npu2)

// CHECK-LABEL: aie.switchbox(%mem_tile_1_1) {
// CHECK-NEXT:   %[[b0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[b1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   aie.masterset(DMA : 3, %[[b1]])
// CHECK-NEXT:   aie.masterset(North : {{[0-9]+}}, %[[b0]])
// CHECK-NEXT:   aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[b1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 5) {
// CHECK-NEXT:     aie.rule(31, 2, %[[b0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Merge tile: both id-0 streams feed one amsel, which drives exactly one port.
// CHECK-LABEL: aie.switchbox(%tile_2_3) {
// CHECK-NEXT:   %[[a0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(South : {{[0-9]+}}, %[[a0]])
// CHECK-NEXT:   aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a0]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a0]])
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

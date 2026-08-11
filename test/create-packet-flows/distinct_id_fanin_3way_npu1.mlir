//===- distinct_id_fanin_3way_npu1.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Control for same_id_fanin_3way_npu1.mlir: identical geometry and identical
// destination, but the three flows carry DISTINCT packet ids 0/1/2.
//
// This pins the behaviour that PR #3472 deliberately kept. Channel sharing is
// the entire point of packet flows, and it stays legal here precisely because
// the ids differ:
//
//   tile_0_3  merges its own DMA:0 (id 1) with the incoming North:1 (id 2)
//             onto a SINGLE amsel and a single link  -- sharing, allowed
//   tile_0_2  splits that shared link back apart BY ID:
//             rule(31, 2) -> its own amsel -> South:3
//             rule(31, 1) -> its own amsel -> South:1
//
// Two same-id flows could not be split like this, which is why the id-gating
// added in #3472 forces them onto disjoint paths instead. Note that no amsel
// drives more than one master port in either test -- that invariant holds
// regardless of whether links are shared.

// CHECK-LABEL: aie.device(npu1)

// (Checks follow the order the switchboxes are emitted in, so the split point
// at tile_0_2 is pinned before the merge point at tile_0_3 even though the data
// flows the other way: tile_0_4 -> tile_0_3 -> tile_0_2 -> shim.)

// The split point: the shared link is demultiplexed by packet id, each id onto
// its own amsel and its own output port.
// CHECK:      %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK:      aie.switchbox(%[[tile_0_2]]) {
// CHECK-NEXT:   %[[a0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[a1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   %[[a2:.*]] = aie.amsel<2> (0)
// CHECK-NEXT:   aie.masterset(South : 0, %[[a0]])
// CHECK-NEXT:   aie.masterset(South : 1, %[[a1]])
// CHECK-NEXT:   aie.masterset(South : 3, %[[a2]])
// CHECK-NEXT:   aie.packet_rules(North : 1) {
// CHECK-NEXT:     aie.rule(31, 2, %[[a2]])
// CHECK-NEXT:     aie.rule(31, 1, %[[a1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// The merge point: two flows with different ids share one amsel and one link.
// CHECK:      %[[tile_0_3:.*]] = aie.tile(0, 3)
// CHECK:      aie.switchbox(%[[tile_0_3]]) {
// CHECK-NEXT:   %[[m:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(South : 1, %[[m]])
// CHECK-NEXT:   aie.packet_rules(North : 1) {
// CHECK-NEXT:     aie.rule(31, 2, %[[m]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[m]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(0, 3)
    %t2 = aie.tile(0, 4)
    aie.packet_flow(0) {
      aie.packet_source<%t0, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%t1, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t2, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
  }
}

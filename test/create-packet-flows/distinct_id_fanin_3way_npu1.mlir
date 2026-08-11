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
//   tile_0_2  merges its own DMA:0 (id 0) with the incoming North:2 (id 2)
//             onto a SINGLE amsel and a single link  -- sharing, allowed
//             (id 1, arriving on North:0, stays on its own amsel/link)
//   tile_0_3  keeps id 1 (its own DMA:0) and id 2 (incoming North:0) apart,
//             each on its own amsel and its own output port
//
// Two same-id flows could not be merged like this, which is why the id-gating
// added in #3472 forces them onto disjoint paths instead. Note that no amsel
// drives more than one master port in either test -- that invariant holds
// regardless of whether links are shared.

// CHECK-LABEL: aie.device(npu1)

// (Checks follow the order the switchboxes are emitted in, so the merge point
// at tile_0_2 is pinned before tile_0_3 even though the data flows the other
// way: tile_0_4 -> tile_0_3 -> tile_0_2 -> shim.)

// The merge point: id 0 (own DMA) and id 2 (incoming) share one amsel and one
// link; id 1 (incoming) stays on its own.
// CHECK:      %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK:      aie.switchbox(%[[tile_0_2]]) {
// CHECK-NEXT:   %[[a0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[a1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   aie.masterset(South : 0, %[[a0]])
// CHECK-NEXT:   aie.masterset(South : 1, %[[a1]])
// CHECK-NEXT:   aie.packet_rules(North : 2) {
// CHECK-NEXT:     aie.rule(31, 2, %[[a0]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(North : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[a1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// The split-apart hop: id 1 (own DMA) and id 2 (incoming) stay on separate
// amsels and separate output ports.
// CHECK:      %[[tile_0_3:.*]] = aie.tile(0, 3)
// CHECK:      aie.switchbox(%[[tile_0_3]]) {
// CHECK-NEXT:   %[[b0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[b1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   aie.masterset(South : 0, %[[b0]])
// CHECK-NEXT:   aie.masterset(South : 2, %[[b1]])
// CHECK-NEXT:   aie.packet_rules(North : 0) {
// CHECK-NEXT:     aie.rule(31, 2, %[[b1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[b0]])
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

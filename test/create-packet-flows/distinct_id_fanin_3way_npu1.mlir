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
// the ids differ: ids that travel together can be told apart again downstream
// by their rules, so they may ride one amsel and one link. Two same-id flows
// could not be, which is why #3472 forces those onto disjoint paths instead.
//
// The checks below follow the order the switchboxes are emitted in, which is
// the shim first and then upward.

// CHECK-LABEL: aie.device(npu1)

// The shim: all three ids converge on a single amsel and a single master port.
// The rule(29, 0) cube covers ids 0 and 2 together -- the pair already merged
// upstream -- and id 1 joins them here.
// CHECK:      %[[shim:.*]] = aie.tile(0, 0)
// CHECK:      aie.switchbox(%[[shim]]) {
// CHECK-NEXT:   %[[sh:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(South : 2, %[[sh]])
// CHECK-NEXT:   aie.packet_rules(North : 0) {
// CHECK-NEXT:     aie.rule(29, 0, %[[sh]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(North : 1) {
// CHECK-NEXT:     aie.rule(31, 1, %[[sh]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// The upstream merge point: id 2 (arriving on North : 2) and id 0 (this tile's
// own DMA) share ONE amsel and ONE master port. id 1 is kept on its own amsel.
// CHECK:      %[[tile_0_2:.*]] = aie.tile(0, 2)
// CHECK:      aie.switchbox(%[[tile_0_2]]) {
// CHECK-NEXT:   %[[shared:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[other:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   aie.masterset(South : 0, %[[shared]])
// CHECK-NEXT:   aie.masterset(South : 1, %[[other]])
// CHECK-NEXT:   aie.packet_rules(North : 2) {
// CHECK-NEXT:     aie.rule(31, 2, %[[shared]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(North : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[other]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[shared]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// No amsel drives more than one master port in either this test or its same-id
// companion; that invariant holds regardless of whether links are shared.

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

//===- packet_meeting_same_id_pairs.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Two sources send id 18 and two send id 15, all four into (0, 0) DMA:1 and
// (0, 4) DMA:1. Each pair has to meet, and so do the two trees they make.
// Once the relax ladder lets a flow meet another id's tree ahead of its
// partner, the first of a pair to route meets the other pair's tree and its
// partner meets it there: all three memtile sources leave on one arbiter, and
// (0, 5)'s id 18 joins them at (0, 4), where the tree turns back down to the
// shim.
// Reduced from router_mutation.py's multi-source mutation of
// router_properties.py npu1 seed 207460168.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[UP:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(North : 0, %[[UP]])
// CHECK:         aie.packet_rules(DMA : 3) {
// CHECK-NEXT:      aie.rule(31, 15, %[[UP]])
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule(31, 15, %[[UP]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 18, %[[UP]])
// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK:         %[[MEET:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 1, %[[MEET]])
// CHECK:         aie.masterset(South : 3, %[[MEET]])
// CHECK:         aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(2, 2, %[[MEET]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 18, %[[MEET]])

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.packet_flow(18) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_4, DMA : 1> }
    aie.packet_flow(15) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_4, DMA : 1> }
  }
}

//===- deterministic_merge_reserves_arbiter.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Arbiter 0 at (1,3) carries a deterministic merge schedule, which grants
// only the slaves it names, and arbiters 1-5 each use msel 0. Flow 11 joins
// an arbiter by a free msel of one of 1-5, never arbiter 0's msels 1-3.

// CHECK-LABEL: aie.switchbox(%tile_1_3)
// CHECK:         %[[NEW:.*]] = aie.amsel<{{[1-5]}}> ({{[1-3]}})
// CHECK:         aie.packet_rules(Core : 0) {
// CHECK-NEXT:      aie.rule(31, 11, %[[NEW]])

module {
  aie.device(npu1) {
    %t13 = aie.tile(1, 3)
    %t12 = aie.tile(1, 2)
    aie.switchbox(%t13) {
      %a0 = aie.amsel<0> (0) deterministic_merge [<North : 1, 1>, <DMA : 0, 1>]
      aie.masterset(South : 1, %a0)
      aie.packet_rules(North : 1) { aie.rule(31, 2, %a0) }
      aie.packet_rules(DMA : 0) { aie.rule(31, 1, %a0) }
      %a1 = aie.amsel<1> (0)
      %a2 = aie.amsel<2> (0)
      %a3 = aie.amsel<3> (0)
      %a4 = aie.amsel<4> (0)
      %a5 = aie.amsel<5> (0)
      aie.masterset(East : 0, %a1)
      aie.masterset(East : 1, %a2)
      aie.masterset(East : 2, %a3)
      aie.masterset(East : 3, %a4)
      aie.masterset(West : 0, %a5)
      aie.packet_rules(North : 2) { aie.rule(31, 20, %a1) }
      aie.packet_rules(North : 3) { aie.rule(31, 21, %a2) }
      aie.packet_rules(West : 1) { aie.rule(31, 22, %a3) }
      aie.packet_rules(West : 2) { aie.rule(31, 23, %a4) }
      aie.packet_rules(East : 1) { aie.rule(31, 24, %a5) }
    }
    aie.packet_flow(11) {
      aie.packet_source<%t13, Core : 0>
      aie.packet_dest<%t12, DMA : 0>
    }
  }
}

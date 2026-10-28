//===- packet_unrelated_flows_share_channel.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Fixed mastersets leave North:5 the only way up from (0, 2). Flows 1 and 2
// share no destination, so the router first keeps them on channels of their
// own; when that fails, it lets them share the one channel left.

// CHECK: aie.switchbox(%tile_0_2) {
// CHECK:   %[[A:.*]] = aie.amsel<5> (0)
// CHECK:   aie.masterset(North : 5, %[[A]])
// CHECK:   aie.packet_rules(South : {{[0-9]+}}) {
// CHECK:     aie.rule(28, 0, %[[A]])

module {
  aie.device(npu1_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %sb_0_2 = aie.switchbox(%t_0_2) {
      %a0 = aie.amsel<0> (0)
      %m0 = aie.masterset(North : 0, %a0)
      %a1 = aie.amsel<1> (0)
      %m1 = aie.masterset(North : 1, %a1)
      %a2 = aie.amsel<2> (0)
      %m2 = aie.masterset(North : 2, %a2)
      %a3 = aie.amsel<3> (0)
      %m3 = aie.masterset(North : 3, %a3)
      %a4 = aie.amsel<4> (0)
      %m4 = aie.masterset(North : 4, %a4)
    }
    aie.packet_flow(1) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_4, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_5, DMA : 0> }
  }
}

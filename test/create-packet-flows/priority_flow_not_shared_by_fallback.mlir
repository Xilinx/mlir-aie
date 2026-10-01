//===- priority_flow_not_shared_by_fallback.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s

// packet_unrelated_flows_share_channel.mlir with flow 2 prioritized. The
// flows may not share North:5 at (0, 2) even when unrelated flows may share:
// a ctrl-packet reload skips the prioritized flow's master set, which would
// leave flow 1 unconfigured.

// CHECK: error: Unable to find a legal routing: packet flows from (0, 1) DMA:1 are prioritized (priority_route), so they keep the route they take alone, and it holds a channel from tile (0, 2) to (0, 3) the other flows need.

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
    aie.packet_flow(2) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_5, DMA : 0> } {priority_route = true}
  }
}

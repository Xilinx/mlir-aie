//===- priority_flow_shared_without_reload.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// packet_unrelated_flows_share_channel.mlir with flow 2 prioritized. Alone,
// flow 2 goes up from (0,2) by a channel the design's own switchbox there
// takes, so without a control-packet reload it routes like the others, and
// the two flows share North:5 at (0,2).

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         %[[A:.*]] = aie.amsel<5> (0)
// CHECK:         aie.masterset(North : 5, %[[A]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(28, 0, %[[A]])

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

// -----

// A control-packet reload keeps the route the control overlay takes alone,
// which goes up from (0,2) by a master port the design's own switchbox there
// takes.

module {
  aie.device(npu1_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
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
    // expected-error@+1 {{Unable to find a legal routing: packet flows from (0, 1) DMA:1 are prioritized (priority_route) in a design a control-packet reload configures (has_ctrl_pkt_overlay), so they keep the route they take alone, as in @ctrl_pkt_overlay, and the other flows route only if it moves. Around it, the route packet flows from (0, 1) DMA:1 take alone does not fit this design: it goes from (0, 2) South:3 to (0, 2) North:0, which the design's own switchboxes leave no connection for.}}
    aie.packet_flow(2) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_1, TileControl : 0> aie.packet_dest<%t_0_2, TileControl : 0> aie.packet_dest<%t_0_3, TileControl : 0> aie.packet_dest<%t_0_4, TileControl : 0> aie.packet_dest<%t_0_5, TileControl : 0> } {priority_route = true}
  } {has_ctrl_pkt_overlay = true}
}

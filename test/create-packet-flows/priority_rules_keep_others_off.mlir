//===- priority_rules_keep_others_off.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows %s | FileCheck %s

// A control-packet reload keeps the master sets and packet rules of the
// prioritized flows (the control overlay), whose rules match first. Another
// flow may take their amsel only where it goes exactly where they do.

// Ids 24-26 take the rule (28, 24), which also matches id 27. Id 27 goes where
// they do, so it follows their route and their rule sends it the right way.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK:         aie.masterset(North : 1, %[[A:.+]]) {aie.is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(28, 24, %[[A]]) {aie.is_ctrl_pkt_overlay, aie.priority_route}
// CHECK-NEXT:      aie.rule(31, 27, %[[A]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK:         aie.masterset(DMA : 0, %[[B:.+]]) {aie.is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(28, 24, %[[B]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:      aie.rule(31, 27, %[[B]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.masterset(North : 1, %[[C:.+]]) {aie.is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(28, 24, %[[C]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT:      aie.rule(31, 27, %[[C]])
// CHECK-NEXT:    }

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_2 = aie.tile(0, 2)
    aie.packet_flow(24) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(25) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(26) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(27) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> }
  }
}

// -----

// Id 27 goes elsewhere, and in a design a control-packet reload configures,
// the overlay's rule (28, 24) would send it where ids 24-26 go.

module {
  // expected-error@+1 {{Unable to find a legal routing: packet flows from (0, 0) DMA:0 are prioritized (priority_route) in a design a control-packet reload configures (has_ctrl_pkt_overlay), so they keep the route they take alone, as in @ctrl_pkt_overlay, and the other flows route only if it moves. Around it, at tile (0, 0), the packet rule (mask 0x1C, id 0x18) of the prioritized flows (the control overlay) on DMA:0 also matches packet id 0x1B, and a control-packet reload keeps their rules; use an id the rule does not match, or route the flow apart.}}
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    aie.packet_flow(24) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_0, TileControl : 0> aie.packet_dest<%t_0_1, TileControl : 0> aie.packet_dest<%t_0_2, TileControl : 0> } {priority_route = true}
    aie.packet_flow(25) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_0, TileControl : 0> aie.packet_dest<%t_0_1, TileControl : 0> aie.packet_dest<%t_0_2, TileControl : 0> } {priority_route = true}
    aie.packet_flow(26) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_0, TileControl : 0> aie.packet_dest<%t_0_1, TileControl : 0> aie.packet_dest<%t_0_2, TileControl : 0> } {priority_route = true}
    aie.packet_flow(27) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> }
  } {has_ctrl_pkt_overlay = true}
}

// -----

// Without a reload, ids 24-26 route like the others, and id 27 takes a rule
// of its own.

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[TO0:[0-9]+]])
// CHECK-DAG:     aie.masterset(DMA : 1, %[[TO1:[0-9]+]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 27, %[[TO1]])
// CHECK-NEXT:      aie.rule(30, 24, %[[TO0]])
// CHECK-NEXT:      aie.rule(31, 26, %[[TO0]])

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_2 = aie.tile(0, 2)
    aie.packet_flow(24) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(25) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(26) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> } {priority_route = true}
    aie.packet_flow(27) { aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> }
  }
}

// -----

// Id 28 reaches memtile (0,1) DMA:2 and DMA:4, and DMA:2 is a master port of
// prioritized id 1. Id 28 routes only if id 1 reaches DMA:2 by a master set
// of its own, which it does not alone. Without a reload, id 1 then routes like
// the others.

// CHECK-LABEL: aie.device(npu2_4col)
// CHECK:       aie.switchbox(%mem_tile_0_1)
// CHECK-NOT:     is_ctrl_pkt_overlay
// CHECK:         aie.masterset(DMA : 2, %[[TWO:[0-9]+]],
// CHECK:         aie.masterset(DMA : 4, %[[FOUR:[0-9]+]])
// CHECK-DAG:     aie.rule(31, 28, %[[FOUR]])
// CHECK-DAG:     aie.rule(31, 1, %[[TWO]])
// CHECK:       aie.switchbox(

module {
  aie.device(npu2_4col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_3_5 = aie.tile(3, 5)
    aie.packet_flow(28) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_source<%t_3_5, Core : 0> aie.packet_dest<%t_0_1, DMA : 2> aie.packet_dest<%t_0_1, DMA : 4> }
    aie.packet_flow(1) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_1, DMA : 2> } {priority_route = true}
  }
}

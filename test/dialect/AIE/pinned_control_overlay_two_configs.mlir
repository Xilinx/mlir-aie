//===- pinned_control_overlay_two_configs.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-pin-control-overlay --aie-create-pathfinder-flows | FileCheck %s
// Pin must have actually run: the control flow carries its captured
// pinned route, an attribute that is absent entirely without
// --aie-pin-control-overlay -- so this guards the feature, not just the
// base pathfinder co-lowering that the CHECK lines below share with it.
// RUN: aie-opt %s --aie-pin-control-overlay --aie-create-pathfinder-flows | FileCheck %s --check-prefix=PINNED
// PINNED: ctrl_pkt_pinned_route

// Task 6 (acceptance gate), Part 2.1: the whole point of pinning is that
// control routing is IDENTICAL across every config device regardless of what
// data that config carries. Two config devices, @cfg_a and @cfg_b, carry
// DIFFERENT data (opposite directions, different packet ids, different
// destination bundles: @cfg_a routes shim->mem_tile->tile(0,2) DMA, @cfg_b
// routes tile(0,2)->mem_tile->shim in the reverse direction on a different
// id). Despite that, both devices' is_ctrl_pkt_overlay-tagged amsel/masterset/
// packet_rules ops -- ports, amsel arbiter/msel values, and rule (mask,
// value) -- are byte-identical to each other AND to @ctrl_pkt_overlay's own
// routing of the same control flow alone.

// Checks below use plain CHECK (not CHECK-NEXT): the per-device pathfinder
// inserts each config's own data switchbox ops at a different position
// relative to the materialized control switchboxes than in @ctrl_pkt_overlay
// (config-specific, harmless), so the control content is matched by its
// distinctive (unique-per-device) amsel/masterset/rule text, not by adjacency
// to a preceding "aie.switchbox" line. What IS invariant, and what these
// checks pin down, is that the SAME two control masters -- (North:3,
// amsel<5>(3)) and (TileControl:0, amsel<5>(3)) -- with the SAME rule (mask
// 31, value 1, is_ctrl_pkt_overlay) appear in @cfg_a, @cfg_b, and
// @ctrl_pkt_overlay, in that relative order, regardless of each config's data.

// CHECK-LABEL: aie.device(npu2) @cfg_a
// (0,0): control master, byte-identical across all three devices below.
// CHECK:   %[[A_CA0:.*]] = aie.amsel<5> (3)
// CHECK:   aie.masterset(North : 3, %[[A_CA0]]) {is_ctrl_pkt_overlay}
// CHECK:   aie.rule(31, 1, %[[A_CA0]]) {is_ctrl_pkt_overlay}
// (0,1): control terminal master, byte-identical across all three devices.
// CHECK:   %[[A_CA1:.*]] = aie.amsel<5> (3)
// CHECK:   aie.masterset(TileControl : 0, %[[A_CA1]]) {is_ctrl_pkt_overlay, keep_pkt_header = true}
// CHECK:   aie.rule(31, 1, %[[A_CA1]]) {is_ctrl_pkt_overlay}

// CHECK-LABEL: aie.device(npu2) @cfg_b
// (0,0): SAME control master as @cfg_a (North:3, amsel<5>(3)), even though
// @cfg_b's data (packet id 9, opposite direction) is entirely different.
// CHECK:   %[[B_CA0:.*]] = aie.amsel<5> (3)
// CHECK:   aie.masterset(North : 3, %[[B_CA0]]) {is_ctrl_pkt_overlay}
// CHECK:   aie.rule(31, 1, %[[B_CA0]]) {is_ctrl_pkt_overlay}
// (0,1): SAME control terminal master as @cfg_a.
// CHECK:   %[[B_CA1:.*]] = aie.amsel<5> (3)
// CHECK:   aie.masterset(TileControl : 0, %[[B_CA1]]) {is_ctrl_pkt_overlay, keep_pkt_header = true}
// CHECK:   aie.rule(31, 1, %[[B_CA1]]) {is_ctrl_pkt_overlay}

// The source of truth: @ctrl_pkt_overlay reproduces the SAME canonical
// control routing both configs were pinned to.
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK:   %[[O_CA0:.*]] = aie.amsel<5> (3)
// CHECK:   aie.masterset(North : 3, %[[O_CA0]]) {is_ctrl_pkt_overlay}
// CHECK:   aie.rule(31, 1, %[[O_CA0]]) {is_ctrl_pkt_overlay}
// CHECK:   %[[O_CA1:.*]] = aie.amsel<5> (3)
// CHECK:   aie.masterset(TileControl : 0, %[[O_CA1]]) {is_ctrl_pkt_overlay, keep_pkt_header = true}
// CHECK:   aie.rule(31, 1, %[[O_CA1]]) {is_ctrl_pkt_overlay}

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    // Control flow (pinned: materialized, decl removed). Identical to @cfg_b's.
    aie.packet_flow(1) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Config A's data: shim -> (0,2) DMA, packet id 7.
    aie.packet_flow(7) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t02, DMA : 0>
    }
  } {sym_name = "cfg_a"}
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    // Control flow: byte-identical to @cfg_a's -- same source/dest/attrs.
    aie.packet_flow(1) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Config B's data: DIFFERENT from A -- reverse direction, different
    // packet id, different source/dest channels.
    aie.packet_flow(9) {
      aie.packet_source<%t02, DMA : 1>
      aie.packet_dest<%t00, DMA : 1>
    }
  } {sym_name = "cfg_b"}
  aie.device(npu2) {
    %o00 = aie.tile(0, 0)
    %o01 = aie.tile(0, 1)
    aie.packet_flow(1) {
      aie.packet_source<%o00, DMA : 0>
      aie.packet_dest<%o01, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}

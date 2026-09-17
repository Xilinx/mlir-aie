//===- freeze_design_aware_coherent.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-freeze-control-fabric="design-aware=true" --aie-create-pathfinder-flows | FileCheck %s

// Design-aware control freeze captures a COHERENT control spine: a control
// multicast leaves its source on a SINGLE output channel, even when config data
// makes weaving onto a fresh channel locally cheaper. The capture routes each
// multicast destination farthest-first, so the longest path establishes the
// column trunk and every nearer destination reuses it (via the trunk discount)
// instead of opening its own channel. A fragmented capture (one channel per
// destination) overflows the source slave port's 4-slot packet-rule cover and
// the design fails to route (the Layer-1-vs-Layer-0 fit regression this fixes).
//
// Fixture: a control spine shim(0,0) -> TileControl at rows 1-4, plus two upward
// circuit data flows occupying the low North channels from row 1 up. A near
// control destination could grab the data-free 1-hop channel while far ones
// detour, splitting the shim into two North masters. Farthest-first keeps the
// shim on a single North master.

// The @cfg shim switchbox drives exactly ONE control master (single coherent
// spine): the control masterset is immediately followed by the packet rules,
// with no second is_ctrl_pkt_overlay master before them.
// CHECK: aie.switchbox(%shim_noc_tile_0_0)
// CHECK: aie.masterset(North : {{[0-9]+}}, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-NOT: aie.masterset({{.*}}) {is_ctrl_pkt_overlay}
// CHECK: aie.packet_rules

module {
  aie.device(npu2) {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    // Control multicast spine up the column: shim -> TileControl at rows 1-4.
    aie.packet_flow(1) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%m01, TileControl : 0>
      aie.packet_dest<%t02, TileControl : 0>
      aie.packet_dest<%t03, TileControl : 0>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Upward circuit data occupying the low North channels from row 1 up, so a
    // near control destination could grab the data-free shim->row1 hop on a
    // channel the far destinations must detour around.
    aie.flow(%m01, DMA : 0, %t04, DMA : 0)
    aie.flow(%m01, DMA : 1, %t04, DMA : 1)
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o00 = aie.tile(0, 0)
    %o01 = aie.tile(0, 1)
    %o02 = aie.tile(0, 2)
    %o03 = aie.tile(0, 3)
    %o04 = aie.tile(0, 4)
    aie.packet_flow(1) {
      aie.packet_source<%o00, DMA : 1>
      aie.packet_dest<%o01, TileControl : 0>
      aie.packet_dest<%o02, TileControl : 0>
      aie.packet_dest<%o03, TileControl : 0>
      aie.packet_dest<%o04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}

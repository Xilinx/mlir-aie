//===- pathfinder_freeze_circuit_avoid.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-freeze-control-fabric --aie-create-pathfinder-flows | FileCheck %s

// View-unification (C) + Layer 0: control is CO-ROUTED and pinned via its
// captured route; Layer 0 then RESERVES control's master ports
// (reservePinnedControlMasters marks their columns connectivity-INVALID) so no
// other flow -- including a circuit flow (aie.flow, lowered to a plain
// aie.connect) -- can be routed onto a control master. A circuit flow that would
// otherwise take a frozen control master's exact (bundle, channel) is routed
// onto a different channel at every hop; the reservation makes that structural
// (edge-absent), not merely demand-discouraged.
//
// The control flow (source (0,1) DMA, dest (0,4) TileControl) pins North:1
// through (0,1)/(0,2). The circuit flow (0,0) DMA:0 -> (0,3) DMA:0 must route
// around it; if it collided on North:1 the switchbox verifier would reject the
// duplicate connect and aie-opt would fail (so the checks below passing at all
// proves no collision), and the exact routed connects pin it off North:1.

// CHECK-LABEL: aie.device(npu2) @cfg
// The frozen control master (North:1, amsel<5>(3), is_ctrl_pkt_overlay) is pinned.
// CHECK-DAG: aie.masterset(North : 1, %{{.*}}) {is_ctrl_pkt_overlay}
// The circuit flow routed around it on North:0 / South:0 at every hop, never
// North:1 (its per-hop connects are pinned below).
// CHECK-DAG: aie.connect<South : 3, North : 0>
// CHECK-DAG: aie.connect<DMA : 0, North : 3>
// CHECK-DAG: aie.connect<South : 0, North : 0>
// CHECK-DAG: aie.connect<South : 0, North : 0>
// CHECK-DAG: aie.connect<South : 0, DMA : 0>
// No circuit connect targets control's North:1 master.
// CHECK-NOT: aie.connect<{{[^>]*}}North : 1>

// The overlay device routes its own control decl normally and reproduces the
// same canonical control master the config pinned to.
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK:        %[[OCA:.*]] = aie.amsel<5> (3)
// CHECK:        aie.masterset(North : 1, %[[OCA]]) {is_ctrl_pkt_overlay}

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    // Control flow (co-routed + pinned). Pins North:1 through (0,1)/(0,2).
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Circuit data flow: through-traffic from the shim toward (0,3) DMA, which
    // (unpinned) would want the same North exit at (0,1)/(0,2) that control uses.
    aie.flow(%t00, DMA : 0, %t03, DMA : 0)
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o01 = aie.tile(0, 1)
    %o02 = aie.tile(0, 2)
    %o03 = aie.tile(0, 3)
    %o04 = aie.tile(0, 4)
    aie.packet_flow(1) {
      aie.packet_source<%o01, DMA : 0>
      aie.packet_dest<%o04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}

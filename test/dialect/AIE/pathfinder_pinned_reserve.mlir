//===- pathfinder_pinned_reserve.mlir --------------------------*- MLIR -*-===//
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

// View-unification (C) + Layer 0: control is CO-ROUTED and pinned; Layer 0 then
// RESERVES control's master ports (connectivity-INVALID on the master column) so
// data cannot be routed onto a control master (the num16 wedge: a config's data
// route repointing a resident control master mid-delivery). Here the pinned
// control master is North:1 through (0,1)/(0,2); the data flow that would
// (unpinned) take North:1 is routed onto a different master instead. If it
// collided on North:1 the switchbox verifier would reject the duplicate master
// and aie-opt would fail, so the checks passing at all proves no collision.
//
// The reservation touches only the master COLUMN, not the slave rows, so
// slave-slot sharing on a SOURCE port is untouched: the two data flows (7, 8)
// that share the shim source still share a slave port on disjoint packet-id
// slots (rules 8 and 7), exactly as in the non-pinning path.

// CHECK-LABEL: aie.device(npu2) @cfg
// (0,1): control's canonical master (North:1, is_ctrl_pkt_overlay) is pinned;
// its DMA:0 slave carries the control rule.
// CHECK:      aie.masterset(North : 1, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK:      aie.packet_rules(DMA : 0) {
// CHECK-NEXT:   aie.rule(31, 1, %{{.*}}) {is_ctrl_pkt_overlay}
// (0,2): the two data flows (8, 7) SHARE slave port South:0 on DISJOINT
// packet-id slots -- the slave-slot-sharing mechanism the pin must not disturb.
// CHECK:      aie.packet_rules(South : 0) {
// CHECK-NEXT:   aie.rule(31, 8, %{{.*}})
// CHECK-NEXT:   aie.rule(31, 7, %{{.*}})
// CHECK-NEXT: }
// The control flow's terminal master (TileControl:0) is pinned at (0,4).
// CHECK:      aie.masterset(TileControl : 0, %{{.*}}) {is_ctrl_pkt_overlay, keep_pkt_header = true}

// The overlay device routes its own control decls normally and reproduces the
// SAME canonical control masters the config pinned to.
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK:      %[[OA:.*]] = aie.amsel<5> (3)
// CHECK:      aie.masterset(North : 1, %[[OA]]) {is_ctrl_pkt_overlay}
// CHECK:      aie.masterset(TileControl : 0, %{{.*}}) {is_ctrl_pkt_overlay, keep_pkt_header = true}

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    // Control flow (co-routed + pinned). Routes North:1 through (0,1)/(0,2) to
    // (0,4) TileControl.
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Data flow: through-traffic from the shim toward a TileControl dest, which
    // (unpinned) would take control's North:1 master. Shares its shim source
    // with flow(8).
    aie.packet_flow(7) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t03, TileControl : 0>
    }
    // Second data flow sharing the shim source with flow(7): exercises
    // slave-slot sharing (disjoint id slots) which the demand-level pin must
    // leave intact.
    aie.packet_flow(8) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, TileControl : 0>
    }
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o00 = aie.tile(0, 0)
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

//===- pathfinder_pinned_merge.mlir ---------------------------*- MLIR -*-===//
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

// View-unification (C): control is CO-ROUTED (pinned), not materialized. When a
// data flow shares the control slave port (here both source tile (0,1) DMA:0),
// addFlow merges the co-sourced legs into one pinned flow: control replays its
// captured route and data is routed around it, rejoining at the shared source
// slave. The native emitClass then lowers control+data onto that ONE shared
// slave port -- control id in the low slot (is_ctrl_pkt_overlay), data id after
// -- for free (the proven non-pinning co-lowering), instead of the (A) path's
// merge-into-a-materialized-op. No second packet_rules is emitted on the slave.

// CHECK-LABEL: aie.device(npu2) @cfg

// (0,1) source switchbox: control's canonical master (North:1, amsel<5>(3),
// is_ctrl_pkt_overlay) is pinned; control (id 1) and data (id 7) SHARE the slave
// DMA:0 in ONE merged packet_rules -- control rule (tagged) in the low slot, the
// data rule appended after.
// CHECK:      aie.switchbox(%{{.*}}) {
// CHECK:        %[[CA:.*]] = aie.amsel<5> (3)
// CHECK:        aie.masterset(North : 1, %[[CA]]) {is_ctrl_pkt_overlay}
// CHECK:        aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 1, %[[CA]]) {is_ctrl_pkt_overlay}
// CHECK-NEXT:     aie.rule(31, 7, %{{.*}})
// CHECK-NEXT:   }
// A second packet_rules on DMA:0 would be a verifier error; there must be exactly
// one on this slave (the merge target).
// CHECK-NOT:    aie.packet_rules(DMA : 0)

// The control packet_flow decl SURVIVES (co-routed), carrying its pinned route.
// CHECK: aie.packet_flow(1)
// CHECK: priority_route = true

// The overlay device routes its control decl normally and reproduces the same
// canonical control master the config pinned to (North:1, amsel<5>(3)).
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK:        %[[OCA:.*]] = aie.amsel<5> (3)
// CHECK:        aie.masterset(North : 1, %[[OCA]]) {is_ctrl_pkt_overlay}

module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    // Control flow (co-routed + pinned).
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Data flow sharing the control slave port (0,1) DMA:0: merges into the
    // pinned flow's shared slave via the native emitClass.
    aie.packet_flow(7) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o01 = aie.tile(0, 1)
    %o02 = aie.tile(0, 2)
    %o03 = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%o01, DMA : 0>
      aie.packet_dest<%o03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}

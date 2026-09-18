//===- coroute_pinned_control.mlir ----------------------------*- MLIR -*-===//
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

// View-unification (C): under pinning the control packet flow is CO-ROUTED, not
// materialized. AIEPinControlOverlay captures @ctrl_pkt_overlay's canonical
// control route and ANNOTATES each config's control packet_flow op with it
// (ctrl_pkt_pinned_route); it no longer clones switch ops or removes the decl.
// The per-device pathfinder then REPLAYS that captured route for the pinned
// control flow (so it cannot drift) while a co-sourced data leg negotiates
// around control's footprint, and the native emitClass merges control+data onto
// the shared slave for free.
//
// Three properties, all of which the (A) materialize path would fail:
//   (a) @cfg's control packet_flow decl is STILL PRESENT (co-routed, not removed).
//   (b) control and data on the shared slave (0,1) DMA:0 land in ONE merged
//       packet_rules -- control id in the low slot (is_ctrl_pkt_overlay), data id
//       appended after.
//   (c) the control master + amsel + rule in @cfg are IDENTICAL to routing the
//       data-free @ctrl_pkt_overlay alone (pinned, no drift).

// CHECK-LABEL: aie.device(npu2) @cfg
// (c) capture the pinned control master port, amsel (arbiter/msel) and rule at
// the source switchbox (0,1).
// CHECK: aie.switchbox(%{{.*}}) {
// CHECK: %[[CA:.*]] = aie.amsel<[[ARB:[0-9]+]]> ([[MSEL:[0-9]+]])
// CHECK: aie.masterset(North : [[CCH:[0-9]+]], %[[CA]]) {is_ctrl_pkt_overlay}
// (b) ONE merged packet_rules on the shared slave: control (tagged) in the low
// slot, data appended after.
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule([[M:[0-9]+]], [[V:[0-9]+]], %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-NEXT: aie.rule({{[0-9]+}}, {{[0-9]+}}, %{{.*}})
// CHECK-NEXT: }
// There must be exactly one packet_rules on this slave (the merge target).
// CHECK-NOT: aie.packet_rules(DMA : 0)
// (a) the control packet_flow declaration survives (co-routed, carrying the
// captured route), printed after the lowered switchboxes.
// CHECK: aie.packet_flow(1)
// CHECK: priority_route = true

// (c) the data-free overlay reproduces the SAME canonical control master port,
// amsel and rule the config pinned to.
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK: aie.amsel<[[ARB]]> ([[MSEL]])
// CHECK: aie.masterset(North : [[CCH]], %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK: aie.rule([[M]], [[V]], %{{.*}}) {is_ctrl_pkt_overlay}

module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    // Control flow (co-routed + pinned; decl NOT removed).
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Data flow sharing the control slave port (0,1) DMA:0: co-sourced, it is
    // merged into the pinned flow, routed around control, and lands in the same
    // packet_rules via the native emitClass.
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

//===- freeze_control_fabric_materialize.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-freeze-control-fabric | FileCheck %s

// View-unification (C): the module-level freeze pass captures @ctrl_pkt_overlay's
// canonical control route ONCE (data-free -> the iteration-0-stable route) and
// ANNOTATES each config's control packet_flow op with it (ctrl_pkt_pinned_route),
// one entry per source. It does NOT materialize switch ops and does NOT remove
// the control decl: control stays a CO-ROUTED packet_flow that the per-device
// pathfinder pins (replays) so it cannot drift. The capture runs a read-only
// routing analysis on the overlay, which is therefore left untouched.

// The config device (printed first): its control packet_flow decls SURVIVE, each
// now carrying a ctrl_pkt_pinned_route annotation alongside its original
// priority_route; its data decl is unchanged; and NO switch ops are materialized
// (the freeze pass no longer lowers anything).
// CHECK-LABEL: aie.device(npu2) @cfg
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%{{.*}}, DMA : 0>
// CHECK:   aie.packet_dest<%{{.*}}, TileControl : 0>
// CHECK: } {ctrl_pkt_pinned_route = {{.*}}, keep_pkt_header = true, priority_route = true}
// CHECK: aie.packet_flow(2) {
// CHECK: } {ctrl_pkt_pinned_route = {{.*}}, keep_pkt_header = true, priority_route = true}
// CHECK: aie.packet_flow(7) {
// CHECK:   aie.packet_source<%{{.*}}, DMA : 0>
// CHECK:   aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK: }
// The freeze pass materializes no switch ops -- control is co-routed, not lowered.
// CHECK-NOT: aie.switchbox
// CHECK-NOT: aie.masterset

// The source-of-truth overlay device is untouched by this pass: it keeps its
// control packet_flow declarations (no annotation) for the per-device pathfinder
// to route.
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK: aie.packet_flow(1)
// CHECK-NOT: ctrl_pkt_pinned_route
// CHECK: priority_route = true

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    // Control flows (co-routed: annotated with the pinned route, decl kept).
    aie.packet_flow(1) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Data flow (kept, routed later by the per-device pathfinder).
    aie.packet_flow(7) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o00 = aie.tile(0, 0)
    %o01 = aie.tile(0, 1)
    %o02 = aie.tile(0, 2)
    aie.packet_flow(1) {
      aie.packet_source<%o00, DMA : 0>
      aie.packet_dest<%o01, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%o00, DMA : 0>
      aie.packet_dest<%o02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}

//===- pinned_adapt.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-pin-control-overlay='mode=blind' --aie-create-pathfinder-flows | FileCheck %s --check-prefix=BLIND
// RUN: aie-opt %s --aie-pin-control-overlay='mode=adapt' --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ADAPT

// Adaptive control pinning (Layer 1, eager avoidance). In blind mode control
// pins with no knowledge of config data (routed on the data-free
// @ctrl_pkt_overlay). With `mode=adapt` (the default) the pinning first routes
// each config's DATA demand and steers the captured control route AROUND it,
// minimizing the overlay's perturbation of an already-routable design.
//
// Fixture: two UPWARD circuit data flows (shim(0,0) -> compute(0,2)) contend
// with the upward control spine for North egress channels at the mem tile (0,1).
// Routed alone, the data takes the two lowest North channels (North:0, North:1).
//   BLIND control captures North:1 (its data-free choice) and DISPLACES a data
//   flow up to North:5 (the design is perturbed).
//   ADAPT control sees the data on North:0/North:1 and detours to North:5, so
//   the data keeps its natural North:0/North:1 routing (zero perturbation), and
//   the @ctrl_pkt_overlay device is pinned to the SAME detour so the resident
//   overlay and the configs' replayed control agree.

// Blind arm: control master on the contended North:1; one data flow bumped to
// North:5.
// BLIND-LABEL: aie.device(npu2) @cfg
// BLIND: aie.switchbox(%mem_tile_0_1)
// BLIND: aie.connect<South : 5, North : 5>
// BLIND: aie.masterset(North : 1, %{{.*}}) {is_ctrl_pkt_overlay}

// Adaptive arm: control master detours to North:5; the data keeps North:1
// (its natural, unperturbed channel).
// ADAPT-LABEL: aie.device(npu2) @cfg
// ADAPT: aie.switchbox(%mem_tile_0_1)
// ADAPT: aie.connect<South : 1, North : 1>
// ADAPT: aie.masterset(North : 5, %{{.*}}) {is_ctrl_pkt_overlay}
// The @ctrl_pkt_overlay device reproduces the SAME detour (resident overlay and
// replayed control agree on the physical port).
// ADAPT-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// ADAPT: aie.masterset(North : 5, %{{.*}}) {is_ctrl_pkt_overlay}

module {
  aie.device(npu2) {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(1) {
      aie.packet_source<%m01, DMA : 0>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Two upward circuit data flows (shim -> compute), same direction as the
    // control spine, contending for North egress at (0,1).
    aie.flow(%s00, DMA : 0, %t02, DMA : 0)
    aie.flow(%s00, DMA : 1, %t02, DMA : 1)
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o01 = aie.tile(0, 1)
    %o02 = aie.tile(0, 2)
    aie.packet_flow(1) {
      aie.packet_source<%o01, DMA : 0>
      aie.packet_dest<%o02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}

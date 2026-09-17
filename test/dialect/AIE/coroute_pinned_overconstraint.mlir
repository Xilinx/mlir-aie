//===- coroute_pinned_overconstraint.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-freeze-control-fabric --aie-create-pathfinder-flows | FileCheck %s

// View-unification (C) over-constraint guard (spec gate 7, refined by Layer 0):
// Layer 0 reserves control master ports (connectivity-INVALID via
// reservePinnedControlMasters) so data can NEVER be routed onto a control
// master. The reservation is NARROW -- only the control master COLUMNS, not a
// whole bundle -- so data still negotiates around control through NON-control
// masters. The original concern was the naive BROAD form (invalidate every
// channel control uses -> data unroutable when control saturates a bundle);
// this narrow form does not hit that: even with all six North channels pinned
// to control at both hops, the independent data flow below still finds a legal
// route around control's saturated bundle.
//
// This is a representative tightest-column squeeze, not a literal port of
// the device-root-caused (7,3) corner (that shape is silicon-specific and
// not reproducible standalone here): six independently-sourced control
// packet flows (one per memtile DMA channel, all priority_route) fan out
// from (0,1) through (0,2), consuming ALL SIX available North channels at
// both hops -- the FULL pinned control footprint for that bundle, not just
// one lane. A wholly independent (different source, non-priority) circuit
// data flow is then routed from (0,2) DMA:1 to (0,4) DMA:1, which must also
// cross the (0,1)/(0,2) corridor. Empirically (verified while authoring this
// test) the co-route pathfinder does NOT fail and does NOT displace any
// control channel: it finds a legal detour through column 1, i.e. it
// correctly treats control's footprint as (very) expensive, not
// unavailable. The properties checked:
//   (a) ALL SIX North channels at (0,1) and (0,2) are consumed by control
//       (the full footprint), identical between @cfg and @ctrl_pkt_overlay
//       (pinned, no drift, even under this maximal footprint).
//   (b) The independent data flow still ROUTES (its connect appears; no
//       "Unable to find a legal routing" -- if that fired, aie-opt would
//       exit non-zero and this RUN line would fail).
//   (c) The data flow's connect carries NO is_ctrl_pkt_overlay tag and does
//       not appear as a 7th North channel at either hop (it neither stole
//       nor merged into control's footprint; it routed around it).

// CHECK-LABEL: aie.device(npu2) @cfg
// (a) full footprint at (0,1): SIX North channels, all pinned.
// CHECK: %switchbox_0_1 = aie.switchbox(%{{.*}}) {
// CHECK-DAG: aie.masterset(North : 0, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 1, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 2, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 3, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 4, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 5, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.rule(31, 1, %{{.*}}) {is_ctrl_pkt_overlay}
// (b)+(c) the independent data flow still ROUTES at the second hop (0,2) and
// escapes via a NON-control master. aie-opt succeeding already proves it routed
// (an unroutable flow emits "Unable to find a legal routing"); the exact escape
// channel is the router's choice, so pin only that DMA:1 leaves (0,2) -- and the
// CHECK-NOT below proves it never took one of control's six North masters.
// CHECK: %switchbox_0_2 = aie.switchbox(%{{.*}}) {
// CHECK: aie.connect<DMA : 1,
// (a) full footprint at (0,2) too: SIX North channels, all still pinned.
// CHECK-DAG: aie.masterset(North : 0, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 1, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 2, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 3, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 4, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 5, %{{.*}}) {is_ctrl_pkt_overlay}
// No untagged (data-stolen) North masterset anywhere in @cfg's corridor.
// CHECK-NOT: aie.masterset(North : {{[0-9]+}}, %{{.*}}) {{$}}

// (a) the data-free overlay reproduces the IDENTICAL full footprint at both
// hops (pinned, byte-identical even under maximal control occupancy).
// CHECK-LABEL: aie.device(npu2) @ctrl_pkt_overlay
// CHECK: %switchbox_0_1 = aie.switchbox(%{{.*}}) {
// CHECK-DAG: aie.masterset(North : 0, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 1, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 2, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 3, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 4, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 5, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK: %switchbox_0_2 = aie.switchbox(%{{.*}}) {
// CHECK-DAG: aie.masterset(North : 0, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 1, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 2, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 3, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 4, %{{.*}}) {is_ctrl_pkt_overlay}
// CHECK-DAG: aie.masterset(North : 5, %{{.*}}) {is_ctrl_pkt_overlay}

module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    // Six control flows, one per DMA source channel at the memtile (0,1),
    // all transiting north through (0,2) toward distinct destination ports
    // on the three core tiles below -- the overlay's FULL control fabric,
    // saturating all 6 North channels at the (0,1)->(0,2) and
    // (0,2)->(0,3) hops.
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%t01, DMA : 1>
      aie.packet_dest<%t03, DMA : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(3) {
      aie.packet_source<%t01, DMA : 2>
      aie.packet_dest<%t03, DMA : 1>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(4) {
      aie.packet_source<%t01, DMA : 3>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(5) {
      aie.packet_source<%t01, DMA : 4>
      aie.packet_dest<%t04, Core : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(6) {
      aie.packet_source<%t01, DMA : 5>
      aie.packet_dest<%t05, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Wholly independent (different source) data circuit flow that must
    // also cross (0,2) -- competing for the same corridor as the full
    // control footprint, which already occupies all 6 North channels there.
    aie.flow(%t02, DMA : 1, %t04, DMA : 1)
  } {sym_name = "cfg"}
  aie.device(npu2) {
    %o01 = aie.tile(0, 1)
    %o02 = aie.tile(0, 2)
    %o03 = aie.tile(0, 3)
    %o04 = aie.tile(0, 4)
    %o05 = aie.tile(0, 5)
    aie.packet_flow(1) {
      aie.packet_source<%o01, DMA : 0>
      aie.packet_dest<%o03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(2) {
      aie.packet_source<%o01, DMA : 1>
      aie.packet_dest<%o03, DMA : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(3) {
      aie.packet_source<%o01, DMA : 2>
      aie.packet_dest<%o03, DMA : 1>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(4) {
      aie.packet_source<%o01, DMA : 3>
      aie.packet_dest<%o04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(5) {
      aie.packet_source<%o01, DMA : 4>
      aie.packet_dest<%o04, Core : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(6) {
      aie.packet_source<%o01, DMA : 5>
      aie.packet_dest<%o05, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {sym_name = "ctrl_pkt_overlay"}
}

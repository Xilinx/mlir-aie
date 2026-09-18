//===- pathfinder_pinned_nonfit.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s --aie-pin-control-overlay --aie-create-pathfinder-flows 2>&1 | FileCheck %s

// Non-fit diagnostic + no-partial-artifact guarantee. An AIE2 mem tile has
// exactly 6 North destination channels (AIETargetModel.cpp
// getNumDestSwitchboxConnections). This config's control flow plus SIX
// independent circuit data flows each need a distinct North egress channel at
// (0,1) toward (0,2) -- seven demands for six channels -- so the routing is
// infeasible. The property under test is the FAILURE MODE: the pathfinder must
// fail with a clean compile-time diagnostic and -- critically -- must NOT emit
// a partially-routed module (a silent wedge waiting to happen on device).
//
// NOTE the over-subscription here is inherent to the channel budget, not the
// pinning reservation specifically: the control flow occupies a North channel at
// (0,1) with or without pinning, so this module fails to route either way. That
// makes it a clean-failure/no-partial-artifact test, NOT a pinning-tradeoff
// test. Pin's tighter cross-device budget (reserving control's master port
// against foreign data) is exercised where it is actually load-bearing, in
// pathfinder_pinned_reserve. The --aie-pin-control-overlay flag is retained
// only to run this diagnostic in a realistic pinning build.

// CHECK: error: Unable to find a legal routing

// No routed artifact reaches stdout on failure (aie-opt's diagnostic goes to
// stderr; both streams are merged into this FileCheck, so absence of any
// wire/connect content here also confirms nothing was printed to stdout
// either -- a successful (even partial) route would emit aie.wire and
// aie.connect ops, which only exist once the pathfinder's emission phase
// runs after findPaths succeeds; it never gets there.
// CHECK-NOT: aie.wire
// CHECK-NOT: aie.connect

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    // Control flow (pinned: co-routed + pinned, decl kept). Layer 0 reserves
    // its North master at (0,1), claiming one of the six North egress channels
    // (and a circuit flow could not share it anyway -- circuit capacity is 1).
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    // Six independent circuit data flows, each needing a DISTINCT dedicated
    // North egress channel at (0,1) to continue toward (0,2). Only 6 total
    // North dest channels exist at a mem tile (AIE2TargetModel); pinning
    // reserves one for control, leaving 5 -- the 6th data flow has no route
    // around the pinned control master.
    aie.flow(%t00, DMA : 0, %t02, DMA : 0)
    aie.flow(%t00, DMA : 1, %t02, DMA : 1)
    aie.flow(%t00, North : 0, %t02, Core : 0)
    aie.flow(%t00, North : 1, %t02, East : 0)
    aie.flow(%t00, North : 2, %t02, West : 0)
    aie.flow(%t00, North : 3, %t02, FIFO : 0)
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

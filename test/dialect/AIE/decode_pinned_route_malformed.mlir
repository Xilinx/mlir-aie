//===- decode_pinned_route_malformed.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aie-create-pathfinder-flows | FileCheck %s

// decodePinnedRoutes hardening: a truncated/garbage ctrl_pkt_pinned_route
// attribute (only reachable via hand-authored IR; encodePinnedRoute always
// emits a well-formed array) must DEGRADE safely -- ignore the pin and route
// the flow normally -- instead of reading past the array. The attribute below
// carries only two i32s where a source header alone needs five, so a bounds-
// unchecked reader would index out of bounds. The malformed pin is ignored (the
// attribute string survives verbatim on the flow op -- it is not stripped -- so
// this is deliberately NOT a `CHECK-NOT: ctrl_pkt_pinned_route`), and the flow
// is routed to its declared destination by the normal pathfinder: assert that
// concrete endpoint so a mis-decoded pin producing a wrong/garbage route
// (rather than crashing) is caught, not just "some switchbox was emitted".

// CHECK: aie.switchbox(%tile_0_3)
// CHECK: aie.masterset(TileControl : 0
module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true,
       ctrl_pkt_pinned_route = [array<i32: 0, 1>]}
  } {sym_name = "cfg"}
}

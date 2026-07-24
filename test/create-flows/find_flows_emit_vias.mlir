//===- find_flows_emit_vias.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// With emit-vias, --aie-find-flows recovers a routed design as grouped flows
// that pin every switchbox hop and then drops the now-redundant switchboxes and
// wires, so re-lowering with --aie-split-flow-vias needs no routing and cannot
// collide with a leftover switchbox.  Route a flow, recover it with vias, split
// it back, and re-route: the switchbox configuration is reproduced.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | FileCheck %s --check-prefix=VIAS --implicit-check-not=aie.switchbox
// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | aie-opt --aie-split-flow-vias --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUTED

// The lifted flow carries the full route and no switchbox survives (the
// --implicit-check-not above rejects any surviving aie.switchbox).
// VIAS: %[[T02:.*]] = aie.tile(0, 2)
// VIAS: %[[T03:.*]] = aie.tile(0, 3)
// VIAS: aie.flow(%[[T02]], DMA : 0, %[[T03]], DMA : 0) via (%[[T02]] : DMA : 0 -> North : {{[0-9]+}}, %[[T03]] : South : {{[0-9]+}} -> DMA : 0)

// ROUTED: %[[T02:.*]] = aie.tile(0, 2)
// ROUTED: aie.switchbox(%[[T02]]) {
// ROUTED:   aie.connect<DMA : 0, North : {{[0-9]+}}>
// ROUTED: %[[T03:.*]] = aie.tile(0, 3)
// ROUTED: aie.switchbox(%[[T03]]) {
// ROUTED:   aie.connect<South : {{[0-9]+}}, DMA : 0>
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
  }
}

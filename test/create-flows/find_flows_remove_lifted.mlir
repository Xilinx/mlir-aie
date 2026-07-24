//===- find_flows_remove_lifted.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// By default --aie-find-flows removes the interconnect configuration it lifts
// into flows, so its output is a purely logical, re-routable design.  With
// remove-lifted=false the switchboxes are kept alongside the recovered flows.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows | FileCheck %s --check-prefix=LIFTED --implicit-check-not=aie.switchbox
// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=remove-lifted=false | FileCheck %s --check-prefix=KEPT
// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows | aie-opt --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUTED

// Lifted: the flow is recovered and every switchbox/wire is gone.
// LIFTED: %[[T02:.*]] = aie.tile(0, 2)
// LIFTED: %[[T03:.*]] = aie.tile(0, 3)
// LIFTED: aie.flow(%[[T02]], DMA : 0, %[[T03]], DMA : 0)

// Kept: switchboxes remain when removal is disabled.
// KEPT: aie.switchbox

// Round-trippable: re-routing the lifted flow reproduces the switchbox config.
// ROUTED: %[[R02:.*]] = aie.tile(0, 2)
// ROUTED: %[[R03:.*]] = aie.tile(0, 3)
// ROUTED: aie.switchbox(%[[R02]]) {
// ROUTED:   aie.connect<DMA : 0, North : {{[0-9]+}}>
// ROUTED: aie.switchbox(%[[R03]]) {
// ROUTED:   aie.connect<South : {{[0-9]+}}, DMA : 0>
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0)
  }
}

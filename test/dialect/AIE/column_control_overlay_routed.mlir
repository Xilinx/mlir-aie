//===- column_control_overlay_routed.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A routed design already carries the overlay if its switchboxes mark a
// TileControl port is_ctrl_pkt_overlay. A priority_route flow between DMAs is
// marked the same way, but it is not the overlay, and skipping the overlay for
// it drops the shim's control response route, which hangs on hardware.

// RUN: aie-opt %s -aie-create-pathfinder-flows | aie-opt -aie-generate-column-control-overlay | FileCheck %s --check-prefix=USER
// RUN: aie-opt %s -aie-generate-column-control-overlay -aie-create-pathfinder-flows | aie-opt -aie-generate-column-control-overlay | FileCheck %s --check-prefix=OVERLAID

// USER:      aie.packet_flow(15) {
// USER-NEXT:   aie.packet_source<%{{.*}}, TileControl : 0>
// USER-NEXT:   aie.packet_dest<%{{.*}}, South : 0>

// OVERLAID:     aie.packet_rules(TileControl : 0) {
// OVERLAID-NOT: aie.packet_flow

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  aie.packet_flow(13) {
    aie.packet_source<%tile_0_1, DMA : 5>
    aie.packet_dest<%tile_0_0, DMA : 1>
  } {priority_route = true}
}

//===- column_control_overlay_idempotent.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The control overlay has to be idempotent, because it is routinely asked to
// run over a design that already carries one. `route-shim-to-tct` defaults to
// `shim-only`, so aiecc lays down the shim's TCT flow on every invocation --
// including for input a caller already overlaid by hand, and including for
// input that declares that flow itself. A second copy is not harmless: it is
// the duplicate `DeviceOp::verify` rejects, so the pass would no longer
// round-trip its own output.

// RUN: aie-opt %s --split-input-file -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" | aie-opt --split-input-file -aie-generate-column-control-overlay | FileCheck %s

// Overlaying twice leaves exactly one TCT flow out of the shim's control port.
// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(15) {
// CHECK-NEXT: aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK-NEXT: }
// CHECK-NOT: aie.packet_dest<%{{.*}}, South : 0>

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2)
}

// -----

// Same, for a design that declares the flow the pass would emit. The ID has to
// match for this to be the same flow, so the tile pins its controller_id.
//
// The surviving flow carries the overlay's attributes even though the input
// declared it without them. keep_pkt_header and priority_route are what mark a
// flow as control to --aie-create-pathfinder-flows, and that pass takes the
// last writer per destination port -- so before this pass deduplicated, the
// copy it appended is what set them. Merely dropping the duplicate would
// change the switchbox this routes into.
// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(3) {
// CHECK-NEXT: aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK-NEXT: } {keep_pkt_header = true, priority_route = true}
// CHECK-NOT: aie.packet_dest<%{{.*}}, South : 0>

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
  %tile_0_2 = aie.tile(0, 2)
  aie.packet_flow(0x3) {
    aie.packet_source<%tile_0_0, TileControl : 0>
    aie.packet_dest<%tile_0_0, South : 0>
  }
}

// -----

// The flow the pass would emit, declared against a *pinned* aie.logical_tile
// rather than the aie.tile. Its coordinates are known, so DeviceOp::verify sees
// it as the same port and would reject a second copy -- matching that means
// recognising the existing flow through TileLike, not just TileOp.
// CHECK-LABEL: module {
// CHECK: %[[shim:.*]] = aie.logical_tile<ShimNOCTile>(0, 0)
// CHECK: aie.packet_flow(3) {
// CHECK-NEXT: aie.packet_source<%[[shim]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[shim]], South : 0>
// CHECK-NEXT: } {keep_pkt_header = true, priority_route = true}
// CHECK-NOT: aie.packet_dest<%{{.*}}, South : 0>

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
  %tile_0_2 = aie.tile(0, 2)
  %shim = aie.logical_tile<ShimNOCTile>(0, 0)
  aie.packet_flow(0x3) {
    aie.packet_source<%shim, TileControl : 0>
    aie.packet_dest<%shim, South : 0>
  }
}

//===- control_overlay_avoid_data.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-control-overlay-avoid-data %s | FileCheck %s --check-prefix=PASS
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=BLIND
// RUN: aie-opt --split-input-file --aie-control-overlay-avoid-data --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=ADAPT

// Two configurations with different data, and the control overlay.
// Routed alone, @a's data leaves (1, 3) on North:3 and @b's leaves (1, 2) on
// North:3. The pass records both ports, with the others the data takes, on
// every device.

// PASS-LABEL: @a
// PASS:       } {ctrl_pkt_avoid = [[AVOID:array<i32: .*1, 2, 5, 3, .*1, 3, 5, 3, .*>]], has_ctrl_pkt_overlay = true}
// PASS-LABEL: @b
// PASS:       } {ctrl_pkt_avoid = [[AVOID]], has_ctrl_pkt_overlay = true}
// PASS-LABEL: @ctrl_pkt_overlay
// PASS:       } {ctrl_pkt_avoid = [[AVOID]], has_ctrl_pkt_overlay = true}

// Routed without regard to the data, the overlay takes both ports, the same
// in every device.

// BLIND-LABEL: @a
// BLIND:       aie.switchbox(%tile_1_2)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// BLIND:       aie.switchbox(%tile_1_3)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// BLIND-LABEL: @b
// BLIND:       aie.switchbox(%tile_1_3)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// BLIND:       aie.switchbox(%tile_1_2)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// BLIND-LABEL: @ctrl_pkt_overlay
// BLIND:       aie.switchbox(%tile_1_2)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// BLIND:       aie.switchbox(%tile_1_3)
// BLIND-NOT:   aie.switchbox
// BLIND:       aie.masterset(North : 3, %{{.+}}) {aie.is_ctrl_pkt_overlay}

// Routed around the data, it takes neither, again the same in every device.
// The router consumes the attribute.

// ADAPT-NOT:   ctrl_pkt_avoid
// ADAPT-LABEL: @a
// ADAPT:       aie.switchbox(%tile_1_2)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 1, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT:       aie.switchbox(%tile_1_3)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 4, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT-LABEL: @b
// ADAPT:       aie.switchbox(%tile_1_3)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 4, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT:       aie.switchbox(%tile_1_2)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 1, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT-LABEL: @ctrl_pkt_overlay
// ADAPT:       aie.switchbox(%tile_1_2)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 1, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT:       aie.switchbox(%tile_1_3)
// ADAPT-NOT:   aie.switchbox
// ADAPT:       aie.masterset(North : 4, %{{.+}}) {aie.is_ctrl_pkt_overlay}
// ADAPT-NOT:   ctrl_pkt_avoid
module {
  aie.device(npu2) @a {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %s10 = aie.tile(1, 0)
    %m11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%s00, TileControl : 0>
      aie.packet_dest<%m01, TileControl : 0>
      aie.packet_dest<%t02, TileControl : 0>
      aie.packet_dest<%t03, TileControl : 0>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(1) {
      aie.packet_source<%s10, DMA : 1>
      aie.packet_dest<%s10, TileControl : 0>
      aie.packet_dest<%m11, TileControl : 0>
      aie.packet_dest<%t12, TileControl : 0>
      aie.packet_dest<%t13, TileControl : 0>
      aie.packet_dest<%t14, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.flow(%s00, DMA : 0, %t14, DMA : 1)
  } {has_ctrl_pkt_overlay = true}
  aie.device(npu2) @b {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %s10 = aie.tile(1, 0)
    %m11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%s00, TileControl : 0>
      aie.packet_dest<%m01, TileControl : 0>
      aie.packet_dest<%t02, TileControl : 0>
      aie.packet_dest<%t03, TileControl : 0>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(1) {
      aie.packet_source<%s10, DMA : 1>
      aie.packet_dest<%s10, TileControl : 0>
      aie.packet_dest<%m11, TileControl : 0>
      aie.packet_dest<%t12, TileControl : 0>
      aie.packet_dest<%t13, TileControl : 0>
      aie.packet_dest<%t14, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.flow(%m11, DMA : 0, %t03, DMA : 0)
  } {has_ctrl_pkt_overlay = true}
  aie.device(npu2) @ctrl_pkt_overlay {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %s10 = aie.tile(1, 0)
    %m11 = aie.tile(1, 1)
    %t12 = aie.tile(1, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%s00, TileControl : 0>
      aie.packet_dest<%m01, TileControl : 0>
      aie.packet_dest<%t02, TileControl : 0>
      aie.packet_dest<%t03, TileControl : 0>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(1) {
      aie.packet_source<%s10, DMA : 1>
      aie.packet_dest<%s10, TileControl : 0>
      aie.packet_dest<%m11, TileControl : 0>
      aie.packet_dest<%t12, TileControl : 0>
      aie.packet_dest<%t13, TileControl : 0>
      aie.packet_dest<%t14, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {has_ctrl_pkt_overlay = true}
}

// -----

// Without @ctrl_pkt_overlay the pass does nothing.

// PASS-LABEL: @alone
// PASS-NOT:   ctrl_pkt_avoid
module {
  aie.device(npu2) @alone {
    %s00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.flow(%s00, DMA : 0, %t02, DMA : 0)
  }
}

// -----

// A design's own prioritized flow is routed with the data, so the overlay
// keeps off its ports too.

// PASS-LABEL: @own_priority
// PASS:       } {ctrl_pkt_avoid = array<i32: 0, 0, 5, 1, 0, 1, 5, 1, 0, 2, 1, 0>, has_ctrl_pkt_overlay = true}
module {
  aie.device(npu2) @own_priority {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%s00, TileControl : 0>
      aie.packet_dest<%m01, TileControl : 0>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(5) {
      aie.packet_source<%s00, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    } {priority_route = true}
  } {has_ctrl_pkt_overlay = true}
  aie.device(npu2) @ctrl_pkt_overlay {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%s00, TileControl : 0>
      aie.packet_dest<%m01, TileControl : 0>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {has_ctrl_pkt_overlay = true}
}

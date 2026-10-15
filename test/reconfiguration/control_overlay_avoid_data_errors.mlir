//===- control_overlay_avoid_data_errors.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-control-overlay-avoid-data --aie-create-pathfinder-flows %s

// The router reads aie.ctrl_pkt_overlay_avoid as a list of #aie.switch_port.

module {
  // expected-error@+1 {{aie.ctrl_pkt_overlay_avoid must list #aie.switch_port attributes}}
  aie.device(npu2) @malformed {
    %s00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {aie.ctrl_pkt_overlay_avoid = [0, 2, 5], has_ctrl_pkt_overlay = true}
}

// -----

// Seven flows leave the memory tile northward, where it has six channels, so
// the data does not route even without the overlay.

module {
  // expected-error@+1 {{the control overlay cannot route around this configuration's data, which does not route without it}}
  aie.device(npu2_1col) @unroutable {
    %s00 = aie.tile(0, 0)
    %m01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.flow(%s00, DMA : 0, %t05, DMA : 0)
    aie.flow(%m01, DMA : 0, %t02, DMA : 0)
    aie.flow(%m01, DMA : 1, %t02, DMA : 1)
    aie.flow(%m01, DMA : 2, %t03, DMA : 0)
    aie.flow(%m01, DMA : 3, %t03, DMA : 1)
    aie.flow(%m01, DMA : 4, %t04, DMA : 0)
    aie.flow(%m01, DMA : 5, %t04, DMA : 1)
  } {has_ctrl_pkt_overlay = true}
  aie.device(npu2_1col) @ctrl_pkt_overlay {
    %s00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%s00, DMA : 1>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {has_ctrl_pkt_overlay = true}
}

//===- overlay_route_alone.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline="builtin.module(aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: aie-opt --pass-pipeline="builtin.module(aie-materialize-runtime-sequences,aie-expand-load-pdi{ctrl-pkt=true})" %t | FileCheck %s --check-prefix=RELOAD
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s

// A design with a control overlay routes the overlay's flows alone,
// as @ctrl_pkt_overlay routes them, even when every flow of the design is
// prioritized. The design's own prioritized flow is not the overlay's, and
// routed with it, it would move the overlay.

// RELOAD: aiex.npu.load_pdi {device_ref = @ctrl_pkt_overlay

// CHECK:      aie.switchbox(%tile_0_2) {
// CHECK:        aie.masterset(DMA : 0, %{{.+}}){{$}}

module {
  aie.device(npu2) @main {
    aie.runtime_sequence @sequence(%arg0 : memref<16xi32>) {
      aiex.configure @design {
      }
    }
  }
  aie.device(npu2) @design {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(5) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
    } {priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%t00, TileControl : 0>
      aie.packet_dest<%t00, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t00, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(26) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t01, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(29) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(30) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(31) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t05, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {has_ctrl_pkt_overlay = true}
  aie.device(npu2) @ctrl_pkt_overlay {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(15) {
      aie.packet_source<%t00, TileControl : 0>
      aie.packet_dest<%t00, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t00, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(26) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t01, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t02, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(29) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t03, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(30) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t04, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(31) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t05, TileControl : 0>
    } {keep_pkt_header = true, priority_route = true}
  } {has_ctrl_pkt_overlay = true}
}

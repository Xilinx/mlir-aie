//===- column_control_overlay_requests_partial.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The response and the requests to the shim and mem tile are routed, but the
// request to the compute tile is not. Only that request is generated, and the
// shim DMA allocation is not repeated.

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" | FileCheck %s

// CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
// CHECK-DAG: %[[COMPUTE:.*]] = aie.tile(0, 2)
// CHECK:     aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CHECK-NOT: aie.shim_dma_allocation
// CHECK:      aie.packet_flow(27) {
// CHECK-NEXT: aie.packet_source<%[[SHIM]], DMA : 0>
// CHECK-NEXT: aie.packet_dest<%[[COMPUTE]], TileControl : 0>
// CHECK-NOT: aie.packet_flow
// CHECK-NOT: aie.shim_dma_allocation

aie.device(npu1_1col) {
  %mem_tile_0_1 = aie.tile(0, 1)
  %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
    %0 = aie.amsel<4> (3)
    %1 = aie.masterset(TileControl : 0, %0) {aie.is_ctrl_pkt_overlay, keep_pkt_header = true}
    aie.packet_rules(South : 4) {
      aie.rule(31, 26, %0)
    } {aie.is_ctrl_pkt_overlay}
  }
  %shim_noc_tile_0_0 = aie.tile(0, 0)
  %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
    aie.connect<DMA : 0, North : 3>
  }
  %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
    %0 = aie.amsel<2> (3)
    %2 = aie.amsel<4> (3)
    %3 = aie.amsel<5> (3)
    %4 = aie.masterset(South : 0, %0) {aie.is_ctrl_pkt_overlay, keep_pkt_header = true}
    %6 = aie.masterset(North : 4, %2) {aie.is_ctrl_pkt_overlay}
    %7 = aie.masterset(TileControl : 0, %3) {aie.is_ctrl_pkt_overlay, keep_pkt_header = true}
    aie.packet_rules(South : 3) {
      aie.rule(31, 26, %2)
      aie.rule(31, 15, %3)
    } {aie.is_ctrl_pkt_overlay}
    aie.packet_rules(TileControl : 0) {
      aie.rule(31, 15, %0)
    } {aie.is_ctrl_pkt_overlay}
  }
  %tile_0_2 = aie.tile(0, 2)
  aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%shim_noc_tile_0_0, MM2S, 0)
}

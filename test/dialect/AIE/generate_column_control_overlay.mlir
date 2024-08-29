//===- generate_column_control_overlay.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay --split-input-file | FileCheck %s
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tct=all-tiles" --split-input-file | FileCheck %s --check-prefix=TCTALLTILES
// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --split-input-file | FileCheck %s --check-prefix=CTRLPKT

// assign controller ids to aie.tile_op, for control packets

// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_0_0]], Ctrl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: } {keep_pkt_header = true}
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0)
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1)
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// CTRLPKT-LABEL: module {
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
// CTRLPKT: memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], Ctrl : 0>
// CTRLPKT: }

aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
}

// -----

// two columns

// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK: %[[tile_1_1:.*]] = aie.tile(1, 1)
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_0_0]], Ctrl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: } {keep_pkt_header = true}
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_1_0]], Ctrl : 0>
// CHECK:   aie.packet_dest<%[[tile_1_0]], South : 0>
// CHECK: } {keep_pkt_header = true}
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0)
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1)
// TCTALLTILES: %[[tile_1_0:.*]] = aie.tile(1, 0)
// TCTALLTILES: %[[tile_1_1:.*]] = aie.tile(1, 1)
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_0]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_1]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// CTRLPKT-LABEL: module {
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CTRLPKT: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CTRLPKT: %[[tile_1_1:.*]] = aie.tile(1, 1)
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
// CTRLPKT: memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_0]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(MM2S, 0, 1)
// CTRLPKT: memref.global "public" @ctrlpkt_col1_mm2s_chan0 : memref<2048xi32>
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_1]], Ctrl : 0>
// CTRLPKT: }

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
}

// -----

// controller_id attribute overriding packet header assignment in aie.packet_flow; 
// round-robin shim dma channel assignment to cover all 5 tiles in a column

// CHECK-LABEL: module {
// CHECK: %[[tile_0_0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CHECK: %[[tile_0_1:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CHECK: %[[tile_0_2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK: %[[tile_0_3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK: %[[tile_0_4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// CHECK: %[[tile_0_5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CHECK: %[[tile_1_0:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK: %[[tile_1_1:.*]] = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// CHECK: aie.packet_flow(4) {
// CHECK:   aie.packet_source<%[[tile_0_0]], Ctrl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: } {keep_pkt_header = true}
// CHECK: aie.packet_flow(5) {
// CHECK:   aie.packet_source<%[[tile_1_0]], Ctrl : 0>
// CHECK:   aie.packet_dest<%[[tile_1_0]], South : 0>
// CHECK: } {keep_pkt_header = true}
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// TCTALLTILES: %[[tile_0_2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// TCTALLTILES: %[[tile_0_3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// TCTALLTILES: %[[tile_0_4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// TCTALLTILES: %[[tile_0_5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// TCTALLTILES: %[[tile_1_0:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// TCTALLTILES: %[[tile_1_1:.*]] = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// TCTALLTILES: aie.packet_flow(4) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(3) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(5) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_2]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(1) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_3]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(6) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_4]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(2) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_5]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(5) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_0]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// TCTALLTILES: aie.packet_flow(7) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_1]], Ctrl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: } {keep_pkt_header = true}
// CTRLPKT-LABEL: module {
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CTRLPKT: %[[tile_0_2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CTRLPKT: %[[tile_0_3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CTRLPKT: %[[tile_0_4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
// CTRLPKT: %[[tile_0_5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
// CTRLPKT: %[[tile_1_0:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CTRLPKT: %[[tile_1_1:.*]] = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// CTRLPKT: aie.packet_flow(4) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
// CTRLPKT: memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
// CTRLPKT: aie.packet_flow(3) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(5) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_2]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(1) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CTRLPKT:   aie.packet_dest<%[[tile_0_3]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1(MM2S, 1, 0)
// CTRLPKT: memref.global "public" @ctrlpkt_col0_mm2s_chan1 : memref<2048xi32>
// CTRLPKT: aie.packet_flow(6) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CTRLPKT:   aie.packet_dest<%[[tile_0_4]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(2) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 1>
// CTRLPKT:   aie.packet_dest<%[[tile_0_5]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(5) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_0]], Ctrl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(MM2S, 0, 1)
// CTRLPKT: memref.global "public" @ctrlpkt_col1_mm2s_chan0 : memref<2048xi32>
// CTRLPKT: aie.packet_flow(7) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_1]], Ctrl : 0>
// CTRLPKT: }

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
  %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
  %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
  %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
  %tile_0_4 = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
  %tile_0_5 = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
  %tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
  %tile_1_1 = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
}

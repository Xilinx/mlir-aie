//===- generate_column_control_overlay.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0)
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1)
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// control routing covers every physical row of the column, not just the
// declared rows 0-1: rows 2-5 get auto-created tiles and control routes too.
// The pass inserts newly-auto-created tiles at the start of the device body,
// so they appear in the emitted IR BEFORE the declared tiles below (in
// descending row order).
// CTRLPKT-LABEL: module {
// CTRLPKT-DAG: %[[tile_0_2:.*]] = aie.tile(0, 2){{$}}
// CTRLPKT-DAG: %[[tile_0_3:.*]] = aie.tile(0, 3) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_0_4:.*]] = aie.tile(0, 4) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_0_5:.*]] = aie.tile(0, 5) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[tile_0_0]], MM2S, 0, <pkt_type = 0, pkt_id = 15>)
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(27) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_2]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(29) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_3]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(30) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_4]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(31) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_5]], TileControl : 0>
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
// CHECK:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_1_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES-LABEL: module {
// TCTALLTILES: %[[tile_0_0:.*]] = aie.tile(0, 0)
// TCTALLTILES: %[[tile_0_1:.*]] = aie.tile(0, 1)
// TCTALLTILES: %[[tile_1_0:.*]] = aie.tile(1, 0)
// TCTALLTILES: %[[tile_1_1:.*]] = aie.tile(1, 1)
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(15) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(26) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// each column's control routing covers every physical row, not just its
// declared rows 0-1. Newly-auto-created tiles are inserted at the start of
// the device body, so (regardless of which column is processed last) all of
// them appear in the emitted IR before any of the declared tiles below.
// CTRLPKT-LABEL: module {
// CTRLPKT-DAG: %[[tile_0_2:.*]] = aie.tile(0, 2){{$}}
// CTRLPKT-DAG: %[[tile_0_3:.*]] = aie.tile(0, 3) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_0_4:.*]] = aie.tile(0, 4) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_0_5:.*]] = aie.tile(0, 5) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_1_2:.*]] = aie.tile(1, 2){{$}}
// CTRLPKT-DAG: %[[tile_1_3:.*]] = aie.tile(1, 3) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_1_4:.*]] = aie.tile(1, 4) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_1_5:.*]] = aie.tile(1, 5) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0)
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1)
// CTRLPKT: %[[tile_1_0:.*]] = aie.tile(1, 0)
// CTRLPKT: %[[tile_1_1:.*]] = aie.tile(1, 1)
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[tile_0_0]], MM2S, 0, <pkt_type = 0, pkt_id = 15>)
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(27) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_2]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(29) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_3]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(30) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_4]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(31) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_5]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(15) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(%[[tile_1_0]], MM2S, 0, <pkt_type = 0, pkt_id = 15>)
// CTRLPKT: aie.packet_flow(26) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_1]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(27) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_2]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(29) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_3]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(30) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_4]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(31) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_5]], TileControl : 0>
// CTRLPKT: }

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
}

// -----

// controller_id attribute overriding packet header assignment in aie.packet_flow;
// single-trunk shim dma channel assignment covers all 5 tiles in a column via
// one shim channel -- rows 3-5 (mandated to the second shim channel by the
// fixed round-robin map) are relocated onto the column's chosen trunk
// (channel 0) and carry ctrl_pkt_shim_chan to record the relocation. Column 1
// declares only rows 0-1; control routing still covers every physical row
// (0-5), so rows 2-5 get auto-created tiles with no manually-assigned
// controller_id. Column 0 already declares all 6 rows, so it gets no new
// tiles; column 1's newly-auto-created tiles are inserted at the start of the
// device body, ahead of every declared tile (including column 0's).

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
// CHECK:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_0_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CHECK: aie.packet_flow(5) {
// CHECK:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// CHECK:   aie.packet_dest<%[[tile_1_0]], South : 0>
// CHECK: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
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
// TCTALLTILES:   aie.packet_source<%[[tile_0_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(3) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(5) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_2]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(1) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_3]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(6) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_4]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(2) {
// TCTALLTILES:   aie.packet_source<%[[tile_0_5]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_0_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(5) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_0]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// TCTALLTILES: aie.packet_flow(7) {
// TCTALLTILES:   aie.packet_source<%[[tile_1_1]], TileControl : 0>
// TCTALLTILES:   aie.packet_dest<%[[tile_1_0]], South : 0>
// TCTALLTILES: }{{.*}}keep_pkt_header = true{{.*}}priority_route = true
// CTRLPKT-LABEL: module {
// CTRLPKT-DAG: %[[tile_1_2:.*]] = aie.tile(1, 2){{$}}
// CTRLPKT-DAG: %[[tile_1_3:.*]] = aie.tile(1, 3) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_1_4:.*]] = aie.tile(1, 4) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT-DAG: %[[tile_1_5:.*]] = aie.tile(1, 5) {ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT: %[[tile_0_0:.*]] = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
// CTRLPKT: %[[tile_0_1:.*]] = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CTRLPKT: %[[tile_0_2:.*]] = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CTRLPKT: %[[tile_0_3:.*]] = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>, ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT: %[[tile_0_4:.*]] = aie.tile(0, 4) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 6>, ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT: %[[tile_0_5:.*]] = aie.tile(0, 5) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 2>, ctrl_pkt_shim_chan = 0 : i32}
// CTRLPKT: %[[tile_1_0:.*]] = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CTRLPKT: %[[tile_1_1:.*]] = aie.tile(1, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
// CTRLPKT: aie.packet_flow(4) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[tile_0_0]], MM2S, 0, <pkt_type = 0, pkt_id = 4>)
// CTRLPKT: aie.packet_flow(3) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_1]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(5) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_2]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(1) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_3]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(6) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_4]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(2) {
// CTRLPKT:   aie.packet_source<%[[tile_0_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_0_5]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(5) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_0]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(%[[tile_1_0]], MM2S, 0, <pkt_type = 0, pkt_id = 5>)
// CTRLPKT: aie.packet_flow(7) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_1]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(27) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_2]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(29) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_3]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(30) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_4]], TileControl : 0>
// CTRLPKT: }
// CTRLPKT: aie.packet_flow(31) {
// CTRLPKT:   aie.packet_source<%[[tile_1_0]], DMA : 0>
// CTRLPKT:   aie.packet_dest<%[[tile_1_5]], TileControl : 0>
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

// -----

// two occupied columns with a gap: flows between column 0 and column 2 route
// through column 1's stream switches, so column 1's switchboxes get configured
// by control packets and need a shim dma allocation of their own. Control
// routing covers every physical row of a covered column (0-5 on npu2), which
// the fixed round-robin map mandates to the second shim channel for some
// rows, but single-trunk selection collapses every covered column onto ONE
// chosen channel regardless of row -- so no column ever needs a second shim
// dma allocation.

// Only the control-packet path covers the gap. Without
// route-shim-to-tile-ctrl the pass runs on every aiecc invocation for every
// target, so it must not reach into a column the design never declared: doing
// so gave such columns routes they previously had none of and left designs
// that used to route with no legal routing at all.
// CHECK-LABEL: module {
// CHECK-NOT: aie.tile(1, {{[0-9]+}})
// CTRLPKT-LABEL: module {
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0
// CTRLPKT-DAG: aie.shim_dma_allocation @ctrlpkt_col2_mm2s_chan0
// CTRLPKT-NOT: mm2s_chan1

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_2_1 = aie.tile(2, 1)
  %tile_2_5 = aie.tile(2, 5)
}

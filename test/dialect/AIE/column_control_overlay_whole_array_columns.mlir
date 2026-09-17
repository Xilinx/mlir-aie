//===- column_control_overlay_whole_array_columns.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" | FileCheck %s

// The route-shim-to-tile-ctrl (control-packet) overlay must cover every column
// of the device, not just the occupied bounding box. The pathfinder routes a
// config's data flows AFTER this pass runs and spills relay switchboxes into
// columns outside the occupied span as a congestion detour; those relays are
// reconfigured by control packets, so control ingress must reach them. This
// design occupies only column 0, but on this 2-column device column 1 must
// still gain its own control ingress DMA and control routes up to the tallest
// occupied row (2). (Companion to column_control_overlay_whole_array.mlir,
// which covers the ROW analogue within the occupied column span.)

// CHECK: aie.device
// The unoccupied column 1 is materialized and covered.
// CHECK-DAG: %[[C1R2:.*]] = aie.tile(1, 2)
// CHECK-DAG: %[[C1R1:.*]] = aie.tile(1, 1)
// CHECK-DAG: %[[C1R0:.*]] = aie.tile(1, 0)
// CHECK: aie.shim_dma_allocation @ctrlpkt_col1_mm2s_chan0(%[[C1R0]], MM2S, 0, <pkt_type = 0, pkt_id = 15>)
// CHECK: aie.packet_flow({{.*}}) {
// CHECK:   aie.packet_source<%[[C1R0]], DMA : 0>
// CHECK:   aie.packet_dest<%[[C1R1]], TileControl : 0>
// CHECK: }
// CHECK: aie.packet_flow({{.*}}) {
// CHECK:   aie.packet_source<%[[C1R0]], DMA : 0>
// CHECK:   aie.packet_dest<%[[C1R2]], TileControl : 0>
// CHECK: }

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
}

//===- column_control_overlay_whole_array.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" | FileCheck %s

// The route-shim-to-tile-ctrl (control-packet) overlay must cover every column
// in the occupied range up to the tallest occupied row -- not just the rows a
// column itself declares. Column 1 here declares only a shim (row 0), but data
// routing spills relay switchboxes into a shim-input / pass-through column's
// upper rows, and those relays are reconfigured by control packets, so control
// must reach them. Column 0 is full-height (rows 0-2), so maxOccupiedRow is 2;
// column 1 must therefore gain control routes to rows 1 and 2.

// CHECK: aie.device
// CHECK-DAG: %[[C1R2:.*]] = aie.tile(1, 2)
// CHECK-DAG: %[[C1R1:.*]] = aie.tile(1, 1)
// CHECK-DAG: %[[C1R0:.*]] = aie.tile(1, 0)
// A control ingress DMA is allocated on the shim-input column, and control is
// routed up to the mem tile (row 1) and the compute tile (row 2).
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
  %tile_1_0 = aie.tile(1, 0)
}

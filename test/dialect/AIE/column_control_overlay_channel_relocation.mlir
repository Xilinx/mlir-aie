//===- column_control_overlay_channel_relocation.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" 2>&1 | FileCheck %s

// A user circuit occupies shim MM2S channel 0 -- the channel the fixed
// round-robin map mandates for row 1. Occupancy-aware selection must relocate
// control ingress to the free channel 1 instead of hard-rejecting: record the
// chosen channel on the column's shim tile (row 0) as `ctrl_pkt_shim_chan = 1`,
// place the control allocation on channel 1, and source the control flow from
// DMA : 1.

// CHECK: aie.tile(0, 0){{.*}}ctrl_pkt_shim_chan = 1
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan1({{.*}}, MM2S, 1, <pkt_type = 0, pkt_id = 15>)
// CHECK: aie.packet_source<{{.*}}, DMA : 1>
// CHECK: aie.packet_dest<{{.*}}tile_0_1{{.*}}, TileControl : 0>

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
}

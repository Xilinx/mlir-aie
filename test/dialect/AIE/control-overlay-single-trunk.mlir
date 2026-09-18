//===- control-overlay-single-trunk.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true whole-array-control-coverage=false" | FileCheck %s

// A column with control tiles at rows 2 and 4 -- under the old fixed
// round-robin map (getRowToShimChanMap) these rows would land on DIFFERENT
// shim MM2S channels (0 and 1 respectively, on npu2's 6-row/2-channel split).
// Single-trunk selection instead collapses the whole column's control onto ONE
// chosen channel (here channel 0, the lowest fully-free channel): every
// packet_source from the column's shim tile names that one channel, and the
// chosen channel is recorded ONCE on the column's shim tile (row 0) as
// `ctrl_pkt_shim_chan`. Control routing covers every physical row of the column
// (0-5 on npu2), not just the declared rows 0/2/4, so row 5 (auto-created,
// not declared) rides the trunk too.

// Positive, order-independent assertions: the single per-column trunk channel
// is recorded on the shim tile (row 0); no controlled tile carries a per-tile
// channel attribute.
// CHECK-DAG: aie.tile(0, 0) {ctrl_pkt_shim_chan = 0 : i32}
// CHECK-DAG: %tile_0_2 = aie.tile(0, 2){{$}}
// CHECK-DAG: %tile_0_3 = aie.tile(0, 3){{$}}
// CHECK-DAG: %tile_0_4 = aie.tile(0, 4){{$}}
// CHECK-DAG: %tile_0_5 = aie.tile(0, 5){{$}}

// The whole column's control ingress allocates exactly one shim channel...
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0({{.*}}, MM2S, 0, <pkt_type = 0, pkt_id = 15>)
// ... and no control packet flow after it sources from any other channel:
// no column emits two distinct shim DMA source channels for control.
// CHECK-NOT: DMA : 1

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_4 = aie.tile(0, 4)
}

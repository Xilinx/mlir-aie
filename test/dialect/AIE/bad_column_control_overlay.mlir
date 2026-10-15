//===- bad_column_control_overlay.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" 2>&1 | FileCheck %s

// Column 1's two shim MM2S channels both move data, so none is left for its
// control packets.

// CHECK: error: 'aie.device' op failed to generate column control overlay: every shim mm2s dma channel of column 1 moves data, so none is left for its control packets

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
  aie.flow(%tile_1_0, DMA : 0, %tile_1_1, DMA : 0)
  aie.packet_flow(1) {
    aie.packet_source<%tile_1_0, DMA : 1>
    aie.packet_dest<%tile_1_1, DMA : 1>
  }
}

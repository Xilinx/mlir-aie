//===- bad_column_control_overlay.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" 2>&1 | FileCheck %s

// Both shim MM2S channels of column 1 are held by circuit-switched flows, so
// control ingress has no channel to relocate to (a circuit monopolizes its
// channel and cannot time-share with control). This is the genuinely
// unsupportable case and must fail with a clear diagnostic. (A single occupied
// channel now relocates to the free one instead of failing -- see
// column_control_overlay_channel_relocation.mlir.)

// CHECK: error: 'aie.device' op failed to generate column control overlay: all shim mm2s dma channels for column 1 are reserved by circuit-switched flows

aie.device(npu2) {
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
  aie.flow(%tile_1_0, DMA : 0, %tile_1_1, DMA : 0)
  aie.flow(%tile_1_0, DMA : 1, %tile_1_1, DMA : 1)
}

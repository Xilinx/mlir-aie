//===- bad_column_control_overlay.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-generate-column-control-overlay="route-shim-to-tile-ctrl=true" --split-input-file 2>&1 | FileCheck %s

// CHECK: error: 'aie.device' op failed to generate column control overlay from shim dma to tile ctrl ports, because some shim mm2s dma channels were reserved from routing control packets.

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
  aie.flow(%tile_1_0, DMA : 0, %tile_1_1, DMA : 0)
}

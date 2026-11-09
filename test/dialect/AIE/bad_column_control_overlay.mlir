//===- bad_column_control_overlay.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s -aie-generate-column-control-overlay="configuration=true " --aie-import-overlay 2>&1 | FileCheck %s

// Column 1's two shim MM2S channels both move data, so none is left for its
// control packets.

// CHECK: failed to generate control overlay: no available shim MM2S channel in column 1

aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_0 = aie.tile(1, 0)
  %tile_1_1 = aie.tile(1, 1)
  aie.flow(%tile_1_0, DMA : 0, %tile_1_1, DMA : 0)
  aie.flow(%tile_1_0, DMA : 1, %tile_1_1, DMA : 1)
}

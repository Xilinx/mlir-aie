//===- bad_channel_out_of_range.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" %s 2>&1 | FileCheck %s

// A pinned channel beyond the tile's DMA channel count is rejected up front.

// CHECK: error: 'aie.objectfifo' op pinned MM2S DMA channel 99 is out of range or already in use on this tile

module @bad_channel_out_of_range {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of (%tile12, {%tile33}, 2 : i32) {prod_dma_channel = 99 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

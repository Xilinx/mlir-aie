//===- bad_channel_collision.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform="dynamic-objFifos=false" --aie-assign-lock-ids %s 2>&1 | FileCheck %s

// Two fifos pin the same MM2S channel on the same producer tile. The second
// reservation collides and is rejected.

// CHECK: error: 'aie.objectfifo' op pinned MM2S DMA channel 0 is out of range or already in use on this tile

module @bad_channel_collision {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile32 = aie.tile(3, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile32}, 2 : i32) {prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile12, {%tile33}, 2 : i32) {prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

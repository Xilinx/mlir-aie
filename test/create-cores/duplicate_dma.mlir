//===- duplicate_dma.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-cores --aie-lower-memcpy %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.dma_start' op duplicate DMA channel MM2S0 not allowed

module @duplicate_dma  {
 aie.device(xcvc1902) {
  %0 = aie.tile(1, 1)
  %1 = aie.buffer(%0) : memref<256xi32>
  %2 = aie.mem(%0)  {
    %15 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
  ^bb1:  // pred: ^bb0
    aiex.useToken @token0(Acquire, 1)
    aie.dma_bd(%1 : memref<256xi32>) { len = 256 : i32 }
    aiex.useToken @token0(Release, 2)
    aie.next_bd ^bb2
  ^bb2:  
    %16 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
  ^bb3:  
    aiex.useToken @token1(Acquire, 1)
    aie.dma_bd(%1 : memref<256xi32>) { len = 256 : i32 }
    aiex.useToken @token1(Release, 2)
    aie.next_bd ^bb4
  ^bb4:  // 4 preds: ^bb0, ^bb1, ^bb2, ^bb3
    aie.end
  }
 }
}
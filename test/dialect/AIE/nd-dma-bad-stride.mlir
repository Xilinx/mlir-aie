//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @tutorial_2b {
  aie.device(xcve2802) {
    %tile14 = aie.tile(1, 4)
    %tile34 = aie.tile(3, 4)

    aie.flow(%tile14, DMA : 0, %tile34, DMA : 0)
    %buf14 = aie.buffer(%tile14) : memref<128xi16>
    %lock14_done = aie.lock(%tile14, 0) { init = 0 : i32 }
    %mem14 = aie.mem(%tile14) {
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op For <32b width datatypes, inner-most dim stride must be 1}}
        aie.dma_bd(%buf14 : memref<128xi16>, 0, 128, [<size = 32, stride = 2>])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
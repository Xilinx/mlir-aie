//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @tutorial_2b {
  aie.device(xcve2802) {
    %tile14 = aie.tile(1, 4)
    %tile34 = aie.tile(3, 4)

    aie.flow(%tile14, DMA : 0, %tile34, DMA : 0)
    aie.buffer @buf14(%tile14) : memref<128xi16>
    aie.lock @lock14_done(%tile14, 0) { init = 0 : i32 }
    %mem14 = aie.mem(%tile14) {
      %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+2 {{'aie.dma_bd' op For <32b width datatypes, inner-most dim stride must be 1}}
        %buf14 = aie.get_buffer @buf14 : memref<128xi16>
        aie.dma_bd(%buf14 : memref<128xi16> offset = 0 len = 128 sizes = [32] strides = [2])
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
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

        %buf14 = aie.buffer(%tile14) { sym_name = "buf14" } : memref<128xi32>

        %lock14_done = aie.lock(%tile14, 0) { init = 0 : i32, sym_name = "lock14_done" }

        %mem14 = aie.mem(%tile14) {
          %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
          ^bd0:
            // The following should generate an out-of-bounds error: the second
            // repetition of accessing array %buf14 with stride of 128 will
            // attempt an access at index 128, which is OOB for a 128xi32 
            // memref.
            // expected-error@+1 {{Specified stride(s) and size(s) result in out of bounds access}}
            aie.dma_bd(%buf14 : memref<128xi32>, dims = [<size = 2, stride = 128>]) { len = 128 : i32 }
            aie.next_bd ^end
          ^end: 
            aie.end
        }

    }
}
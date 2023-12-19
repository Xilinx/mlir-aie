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
    
    AIE.device(xcve2802) {
        %tile14 = AIE.tile(1, 4)
        %tile34 = AIE.tile(3, 4)

        AIE.flow(%tile14, DMA : 0, %tile34, DMA : 0)

        %buf14 = AIE.buffer(%tile14) { sym_name = "buf14" } : memref<128xi32>

        %lock14_done = AIE.lock(%tile14, 0) { init = 0 : i32, sym_name = "lock14_done" }

        %mem14 = AIE.mem(%tile14) {
          %srcDma = AIE.dma_start("MM2S", 0, ^bd0, ^end)
          ^bd0:
            // The following should generate an out-of-bounds error: the second
            // repetition of accessing array %buf14 with stride of 128 will
            // attempt an access at index 128, which is OOB for a 128xi32 
            // memref.
            // expected-error@+1 {{Specified stepsize(s) and wrap(s) result in out of bounds access}}
            AIE.dma_bd(<%buf14 : memref<128xi32>, 0, 128>, A, [<2, 128>])
            AIE.next_bd ^end
          ^end: 
            AIE.end
        }

    }
}
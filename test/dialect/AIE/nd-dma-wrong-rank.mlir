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

        %buf14 = AIE.buffer(%tile14) { sym_name = "buf14" } : memref<3x2x64xi32>

        %lock14_done = AIE.lock(%tile14, 0) { init = 0 : i32, sym_name = "lock14_done" }

        %mem14 = AIE.mem(%tile14) {
          %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
          ^bd0:
            // Currently, we only allow multi-dimensional stride/wrap definitons
            // on BDs referring to a memref of rank 1.
            // expected-error@+1 {{Specifying transfer step sizes and wraps is only supported for }}
            AIE.dmaBd(<%buf14 : memref<3x2x64xi32>, 0, 128>, 0, [<2, 8>])
            AIE.nextBd ^end
          ^end: 
            AIE.end
        }

    }
}
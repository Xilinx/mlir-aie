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

        %buf14 = AIE.buffer(%tile14) { sym_name = "buf14" } : memref<128xi8>

        %lock14_done = AIE.lock(%tile14, 0) { init = 0 : i32, sym_name = "lock14_done" }

        %mem14 = AIE.mem(%tile14) {
          %srcDma = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
          ^bd0:
            // We should ensure multi-dimensional strides/wraps are only used
            // on i32 data types, because that is the elemental transfer size
            // the hardware does.
            // Cast to an i32 memref type if needed.
            //expected-error@+1 {{Specifying transfer step sizes and wraps is only supported for }}
            AIE.dmaBd(<%buf14 : memref<128xi8>, 0, 128>, 0, [<8, 2>])
            AIE.nextBd ^end
          ^end: 
            AIE.end
        }

    }
}
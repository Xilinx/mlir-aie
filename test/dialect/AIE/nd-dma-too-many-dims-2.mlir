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
        %tile33 = aie.tile(3, 3)

        %buf33 = aie.buffer(%tile33) { sym_name = "buf33" } : memref<128xi32>

        %mem33 = aie.mem(%tile33) {
          %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
          ^bd0:
            //expected-error@+1 {{Cannot give more than 3 dimensions}}
            aie.dma_bd(%buf33 : memref<128xi32>, 0, 128, [<size = 1, stride = 1>, <size = 1, stride = 1>, <size = 1, stride = 1>, <size = 1, stride = 1>])
            aie.next_bd ^end
          ^end: 
            aie.end
        }

    }
}
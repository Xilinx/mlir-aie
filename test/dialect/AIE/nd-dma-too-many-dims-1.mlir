//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module @tutorial_2b {
    
    aie.device(xcve2802) {
        %tile31 = aie.tile(3, 1)

        aie.buffer @buf31(%tile31) : memref<128xi32>

        %mem31 = aie.memtile_dma(%tile31) {
          %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
          ^bd0:
            //expected-error@+2 {{Cannot give more than 4 dimensions}}
            %buf31 = aie.get_buffer @buf31 : memref<128xi32>
            aie.dma_bd(%buf31 : memref<128xi32> offset = 0 len = 128 sizes = [1, 1, 1, 1, 1] strides = [1, 1, 1, 1, 1])
            aie.next_bd ^end
          ^end: 
            aie.end
        }

    }
}
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

        %buf31 = aie.buffer(%tile31) { sym_name = "buf31" } : memref<128xi32>

        %mem31 = aie.memtile_dma(%tile31) {
          %c0_i32 = arith.constant 0 : i32
          %c128_i32 = arith.constant 128 : i32
          %srcDma = aie.dma_start("MM2S", 0, ^bd0, ^end)
          ^bd0:
            //expected-error@+1 {{Cannot give more than 4 dimensions}}
            aie.dma_bd(%buf31 : memref<128xi32> offset = %c0_i32 len = %c128_i32 sizes = [1, 1, 1, 1, 1] strides = [1, 1, 1, 1, 1])
            aie.next_bd ^end
          ^end: 
            aie.end
        }

    }
}
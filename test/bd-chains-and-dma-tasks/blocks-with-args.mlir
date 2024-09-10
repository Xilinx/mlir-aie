//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

// RUN: aie-opt --aie-substitute-shim-dma-allocations %s

// Region simplification lead to compiler crashes previously if blocks/regions
// had input arguments, such as created by scf.for loops with yield statements.
// This test ensures that the compiler does not crash with such IR present.

module {
  aie.device(npu1_4col) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) : memref<32xi32>
    %buf0 = aie.buffer(%tile_0_2) : memref<32xi32>
    %buf1 = aie.buffer(%tile_0_2) : memref<32xi32>
    aie.core(%tile_0_2) {
        %c0 = index.constant 0
        %c2 = index.constant 2
        %val_i32 = memref.load %buf[%c0] : memref<32xi32>
        %val = index.castu %val_i32 : i32 to index
        %val_mod_2 = index.rems %val, %c2
        %cond = index.cmp eq(%val_mod_2, %c0)
        %res = scf.if %cond -> (memref<32xi32>) {
            scf.yield %buf0 : memref<32xi32>
        } else {
            scf.yield %buf1 : memref<32xi32>
        }
        %res_load = memref.load %res[%c0] : memref<32xi32>
        memref.store %res_load, %buf[%c0] : memref<32xi32> 
        aie.end
    }
  }
}

//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-assign-buffer-addresses=basic-alloc=0 %s

module {
  aie.device(npu1_2col) {
    // expected-error@+2 {{allocated buffers exceeded available memory}}
    // expected-note@+1 {{}}
    %tile_0_2 = aie.tile(0, 2)
    %C_L1L2_0_0_buff_0 = aie.buffer(%tile_0_2) : memref<64x96xf32> 
    %C_L1L2_0_0_buff_1 = aie.buffer(%tile_0_2) : memref<64x96xf32> 
    %B_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) : memref<32x96xbf16> 
    %B_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) : memref<32x96xbf16> 
    %A_L2L1_0_0_cons_buff_0 = aie.buffer(%tile_0_2) : memref<64x32xbf16> 
    %A_L2L1_0_0_cons_buff_1 = aie.buffer(%tile_0_2) : memref<64x32xbf16> 
  }
}
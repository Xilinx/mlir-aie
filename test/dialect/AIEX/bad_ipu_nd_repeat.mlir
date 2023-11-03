//===- bad_ipu_nd_repeat.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics %s

module {
  AIE.device(ipu) {
    func.func @sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %c16 = arith.constant 16 : i32
      %c32 = arith.constant 32 : i32
      %c128 = arith.constant 128 : i32
      // expected-error@+1 {{Length 3 exceeds the [1:64] range}}
      AIEX.ipu.dma_memcpy_nd (%c0, %c0, %in[%c0,%c0,%c0,%c0][%c128,%c2,%c2,%c8][%c0,%c16,%c8]) { metadata = @of_fromMem, id = 0 : i32 } : (i32, i32, memref<128x4x2x8xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
      return
    }
    AIE.shimDMAAllocation @of_fromMem (MM2S, 0, 0)
  }
}
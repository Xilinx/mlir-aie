//===- shim_AIE2_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 3rd 2023
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-dma-to-npu %s | FileCheck %s

// CHECK:    error: 'aiex.npu.dma_memcpy_nd' op Maximum element bit width allowed is 32 bits.


module @shimDmaMemcpy{
  aie.device(xcve2302) {
    memref.global "public" @toMem : memref<65536xi64>
    func.func @sequence(%arg0: memref<65536xi64>, %arg1: memref<65536xi64>, %arg2: memref<65536xi64>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][4, 4, 64, 64][0, 64, 256]) {id = 0 : i64, metadata = @toMem} : memref<65536xi64>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}


//===- bad_dma_to_npu_datatype.mlir --------------------------*- MLIR -*-===//
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

// RUN: not aie-opt --aie-dma-to-npu %s 2>&1 | FileCheck %s

// CHECK:    error: 'aiex.npu.dma_memcpy_nd' op Maximum element bit width allowed is 32bits.


module @shimDmaMemcpy{
  aie.device(xcve2302) {
    aie.runtime_sequence(%arg0: memref<65536xi64>, %arg1: memref<65536xi64>, %arg2: memref<65536xi64>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][4, 4, 64, 64][0, 64, 256, 1]) {id = 0 : i64, metadata = @toMem} : memref<65536xi64>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}


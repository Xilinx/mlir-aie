//===- aiex_standard_lowering.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aiex-standard-lowering %s | FileCheck %s

// CHECK-LABEL: dma_and_wait
// CHECK-NOT: aiex.npu.dma_memcpy_nd
// CHECK-NOT: aiex.npu.dma_wait
module  {
  aie.device(npu) {
    memref.global "public" @toMem : memref<16xi32>
    func.func @dma_and_wait(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @toMem}
      return
    }
    aie.shim_dma_allocation @toMem (MM2S, 1, 1)
  }
}

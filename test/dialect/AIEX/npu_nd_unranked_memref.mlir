//===- npu_nd_unranked_memref.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Unranked memref support.

// RUN: aie-opt %s

module {
  aie.device(npu1) {
    aie.shim_dma_allocation @airMemcpyId9(MM2S, 0, 0)
    aie.runtime_sequence @bare_matmul(%arg0: memref<*xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 4, 512][0, 0, 512, 1]) {id = 0 : i64, metadata = @airMemcpyId9} : memref<*xf32>
    }
  }
}

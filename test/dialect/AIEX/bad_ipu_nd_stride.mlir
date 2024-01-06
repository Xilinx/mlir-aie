//===- bad_ipu_nd_stride.mlir ----------------------------------*- MLIR -*-===//
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
  aie.device(ipu) {
    func.func @sequence(%in : memref<8388608xi32>, %buf : memref<32xi32>, %out : memref<8388608xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c2 = arith.constant 2 : i64
      %c2097152 = arith.constant 2097152 : i64
      // expected-error@+1 {{Stride 1 exceeds the [1:1M] range}}
      aiex.ipu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c2,%c2][%c0,%c0,%c2097152]) { metadata = @of_fromMem, id = 0 : i64 } : memref<8388608xi32>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}
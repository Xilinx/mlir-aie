//===- bad_ipu_nd_type.mlir ------------------------------------*- MLIR -*-===//
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
    func.func @sequence(%in : memref<1920x1080xi8>, %buf : memref<32xi32>, %out : memref<1920x1080xi8>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c1920 = arith.constant 1920 : i64
      %c1080 = arith.constant 1080 : i64
      // expected-error@+1 {{must be used with memref type with element width 32.}}
      aiex.ipu.dma_memcpy_nd (0, 0, %in[%c0,%c0,%c0,%c0][%c1,%c1,%c1080,%c1920][%c0,%c0,%c1920]) { metadata = @of_fromMem, id = 0 : i64 } : memref<1920x1080xi8>
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}
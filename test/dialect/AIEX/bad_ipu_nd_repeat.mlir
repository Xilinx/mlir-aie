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
  aie.device(ipu) {
    func.func @sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{Length 3 exceeds the [1:64] range}}
      aiex.ipu.dma_memcpy_nd(0, 0, %in : memref<128x4x2x8xi32>) { offsets = array<i32: 0, 0, 0, 0>, lengths = array<i32: 128, 2, 2, 8>, strides = array<i32: 0, 16, 8>,  metadata = @of_fromMem, id = 0 : i32 }
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}
//===- dma_to_npu_invalid.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu --verify-diagnostics %s

module  {
  aie.device(npu) {
    memref.global "public" @toMem : memref<16xi32>
    func.func @sequence() {
      // expected-error@+2 {{failed to legalize operation 'aiex.npu.dma_wait' that was explicitly marked illegal}}
      // expected-error@+1 {{couldn't find shim_dma_allocation op}}
      aiex.npu.dma_wait {symbol = @toMem}
      return
    }
  }
}

//===- invalid.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

func.func @ipu_dma_wait_no_device() {
  // expected-error@+1 {{'aiex.ipu.dma_wait' op couldn't find parent of type DeviceOp}}
  aiex.ipu.dma_wait {symbol = @out0}
  return
}

// -----

aie.device(ipu) {
  func.func @ipu_dma_wait_no_symbol() {
    // expected-error@+1 {{'aiex.ipu.dma_wait' op couldn't find symbol in parent device}}
    aiex.ipu.dma_wait {symbol = @out0}
    return
  }
}

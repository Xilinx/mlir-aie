//===- invalid.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  aie.runtime_sequence() {
    // expected-error@+1 {{'aiex.npu.dma_wait' op couldn't find symbol in parent device}}
    aiex.npu.dma_wait {symbol = @out0}
  }
}

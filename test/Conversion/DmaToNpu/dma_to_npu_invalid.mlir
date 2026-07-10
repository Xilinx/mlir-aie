//===- dma_to_npu_invalid.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu --verify-diagnostics %s

module  {
  aie.device(npu1) {
    aie.runtime_sequence() {
      // expected-error@+1 {{couldn't find symbol in parent device}}
      aiex.npu.dma_wait {symbol = @toMem}
    }
  }
}

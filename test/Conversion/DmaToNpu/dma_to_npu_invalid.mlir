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
  aie.device(npu1) {
    aie.runtime_sequence() {
      // expected-error@+1 {{couldn't find symbol in parent device}}
      aiex.npu.dma_wait {symbol = @toMem}
    }
  }
}

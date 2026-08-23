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

// -----

// set_lock takes a runtime_sequence ANCESTOR, not necessarily a parent -- a
// rolled loop or a select arm may hold one. It still has to be under a
// sequence somewhere, though: this one is under a plain func.
func.func @set_lock_outside_sequence() {
  %tile22 = aie.tile(2, 2)
  %lock22_0 = aie.lock(%tile22, 0) {init = 0 : i32}
  // expected-error@+1 {{'aiex.set_lock' op expects ancestor op 'aie.runtime_sequence'}}
  aiex.set_lock(%lock22_0, 1)
  return
}

//===- bad_ipu_push_queue_repeat.mlir --------------------------*- MLIR -*-===//
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
      %of_fromMem = aie.shim_dma_allocation(MM2S, 0, 0)
      // expected-error@+1 {{Repeat count exceeds the [0:255] range.}}
      aiex.ipu.shimtile_push_queue(%of_fromMem) {issue_token = false, repeat_count = 384 : i32, bd_id = 8 : i32 }
      return
    }
  }
}
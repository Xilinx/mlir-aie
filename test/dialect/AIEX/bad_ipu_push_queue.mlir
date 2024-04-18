//===- bad_npu_push_queue_bd.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu) {
    func.func @bad_bd_id(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{BD ID exceeds the maximum ID.}}
      aiex.npu.shimtile_push_queue {metadata = @of_fromMem, issue_token = false, repeat_count = 3 : i32, bd_id = 28 : i32 }
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}

// -----

module {
  aie.device(npu) {
    func.func @bad_repeat_count(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{Repeat count exceeds the [0:255] range.}}
      aiex.npu.shimtile_push_queue {metadata = @of_fromMem, issue_token = false, repeat_count = 384 : i32, bd_id = 8 : i32 }
      return
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
  }
}
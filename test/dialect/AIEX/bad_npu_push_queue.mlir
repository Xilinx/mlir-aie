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
  aie.device(npu1_4col) {
    aiex.runtime_sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{BD ID exceeds the maximum ID.}}
      aiex.npu.push_queue (0, 0, MM2S:0) {issue_token = false, repeat_count = 3 : i32, bd_id = 28 : i32 }
    }
  }
}

// -----

module {
  aie.device(npu1_4col) {
    aiex.runtime_sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      // expected-error@+1 {{Repeat count exceeds the [0:255] range.}}
      aiex.npu.push_queue (0, 0, MM2S:0) {issue_token = false, repeat_count = 384 : i32, bd_id = 8 : i32 }
    }
  }
}
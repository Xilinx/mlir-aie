//===- bad_npu_push_queue_bd.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      %rc = arith.constant 3 : i32
      %bd = arith.constant 28 : i32
      // expected-error@+1 {{BD ID exceeds the maximum ID.}}
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %bd repeat %rc {issue_token = false} : i32, i32
    }
  }
}


// -----

module {
  aie.device(npu1) {
    aie.runtime_sequence(%in : memref<128x4x2x8xi32>, %buf : memref<32xi32>, %out : memref<8192xi32>) {
      %rc = arith.constant 384 : i32
      %bd = arith.constant 8 : i32
      // expected-error@+1 {{Repeat count exceeds the [0:255] range.}}
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %bd repeat %rc {issue_token = false} : i32, i32
    }
  }
}
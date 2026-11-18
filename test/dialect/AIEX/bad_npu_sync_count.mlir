//===- bad_npu_sync_count.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A sync waits on at least one column and one row.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu2) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-error@+1 {{column_num must cover at least one, got 0}}
    aiex.npu.sync(%c0, %c0, %c0, %c0, %c0, %c1) : i32, i32, i32, i32, i32, i32
  }
}

// -----

aie.device(npu2) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // expected-error@+1 {{row_num must cover at least one, got 0}}
    aiex.npu.sync(%c0, %c0, %c0, %c0, %c1, %c0) : i32, i32, i32, i32, i32, i32
  }
}

//===- require_canonicalize.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s

// A constraint proven true at compile time is erased, so a fully specialized
// sequence carries no runtime check; a constraint on a runtime scalar stays.

// CHECK-LABEL: aie.runtime_sequence @seq
// CHECK-SAME: (%[[BUF:.*]]: memref<32xi32>, %[[N:.*]]: i32)
// CHECK-NOT: aiex.npu.require(%true)
// CHECK: %[[OK:.*]] = arith.cmpi sgt, %[[N]]
// CHECK-NEXT: aiex.npu.require(%[[OK]]) {message = "n must be positive"} : i1
// A repeat of the same condition in the block is dropped, whatever its text.
// CHECK-NOT: aiex.npu.require
module {
  aie.device(npu1) {
    aie.runtime_sequence @seq(%buf : memref<32xi32>, %n : i32) {
      %true = arith.constant true
      aiex.npu.require(%true) {message = "always holds"} : i1
      %c0 = arith.constant 0 : i32
      %ok = arith.cmpi sgt, %n, %c0 : i32
      aiex.npu.require(%ok) {message = "n must be positive"} : i1
      aiex.npu.require(%ok) {message = "n must be positive (again)"} : i1
    }
  }
}

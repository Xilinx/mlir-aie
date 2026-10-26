//===- assert_fold.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-cpp %s | FileCheck %s

// The C++ builder folds the sequence first: a constraint proven true at
// compile time is erased, so a fully specialized sequence carries no runtime
// check, while a constraint on a runtime scalar stays.

// CHECK-LABEL: generate_txn_main_seq
// CHECK-NOT: always holds
// CHECK: return aie_runtime::txn_refused("n must be positive");
// CHECK-NOT: txn_refused
// CHECK: return std::move(txn);
module {
  aie.device(npu1) {
    aie.runtime_sequence @seq(%buf : memref<32xi32>, %n : i32) {
      %true = arith.constant true
      cf.assert %true, "always holds"
      %c0 = arith.constant 0 : i32
      %ok = arith.cmpi sgt, %n, %c0 : i32
      cf.assert %ok, "n must be positive"
      %addr = arith.constant 119300 : i32
      aiex.npu.write32(%addr, %n) : i32, i32
    }
  }
}

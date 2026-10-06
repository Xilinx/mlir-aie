//===- npu_assert_error.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The static TXN binary has no way to refuse a dispatch, so a cf.assert must
// fold away before it gets there. A constant-false one is a compile-time error
// (the same outcome as the ValueError the static Python path raises), and one
// on a runtime value points at the C++ builder. A constant-true one is dropped.

// RUN: not aie-translate --aie-npu-to-binary -aie-output-binary=false --split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'cf.assert' op is violated at compile time: A must tile into (m*rows, k) blocks
module {
  aie.device(npu1) {
    aie.runtime_sequence @live(%buf : memref<32xi32>) {
      %false = arith.constant false
      cf.assert %false, "A must tile into (m*rows, k) blocks"
    }
  }
}

// -----

// CHECK: 'cf.assert' op runtime check cannot be encoded in a static TXN binary; use the C++ builder (--aie-npu-to-cpp) or specialize the value: n must be positive
module {
  aie.device(npu1) {
    aie.runtime_sequence @runtime(%buf : memref<32xi32>, %n : i32) {
      %c0 = arith.constant 0 : i32
      %ok = arith.cmpi sgt, %n, %c0 : i32
      cf.assert %ok, "n must be positive"
    }
  }
}

// -----

// CHECK-NOT: always holds
// CHECK: 'aiex.npu.write32' op Cannot translate write32
module {
  aie.device(npu1) {
    aie.runtime_sequence @holds(%n : i32) {
      %true = arith.constant true
      cf.assert %true, "always holds"
      %addr = arith.constant 100 : i32
      aiex.npu.write32(%addr, %n) : i32, i32
    }
  }
}

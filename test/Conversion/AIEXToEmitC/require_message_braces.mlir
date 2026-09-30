//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s

// A guard message that quotes a value's IR contains braces, which
// emitc.verbatim would read as placeholders; they must reach the C++ string
// literal verbatim, and a line break must not break it.

// CHECK: if (!({{v[0-9]+}})) return aie_runtime::txn_refused("sizes must be >= 1, got [{v = 3 : i32}, 1] on one line");
// CHECK: aie_runtime::txn_append_write32(txn,
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %n: i32) {
      %c1 = arith.constant 1 : i32
      %ok = arith.cmpi sge, %n, %c1 : i32
      aiex.npu.require(%ok) {message = "sizes must be >= 1, got [{v = 3 : i32}, 1]\0Aon one line"} : i1
      %addr = arith.constant 100 : i32
      aiex.npu.write32(%addr, %n) : i32, i32
    }
  }
}

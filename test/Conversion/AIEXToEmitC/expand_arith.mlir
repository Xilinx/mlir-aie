//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s

// A staged scalar computed with `//`, ceildiv, min and max: the Python
// bindings emit arith.floordivsi / ceildivsi / minsi / maxsi, which
// ArithToEmitC has no patterns for. They must expand to the divsi / cmpi /
// select forms it lowers, and the generated C++ must contain no arith op.

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[N:v[0-9]+]], int32_t [[S:v[0-9]+]]) {
// CHECK-NOT: arith
// CHECK: {{v[0-9]+}} / {{v[0-9]+}};
// CHECK: ? {{v[0-9]+}} : {{v[0-9]+}};
// CHECK: aie_runtime::txn_append_write32(txn,
// CHECK: return std::move(txn);
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %n: i32, %start: i32) {
      %c64 = arith.constant 64 : i32
      %c1 = arith.constant 1 : i32
      %q = arith.floordivsi %n, %c64 : i32
      %r = arith.ceildivsi %n, %c64 : i32
      %lo = arith.maxsi %q, %c1 : i32
      %hi = arith.minsi %r, %start : i32
      %v = arith.muli %lo, %hi : i32
      %addr = arith.constant 100 : i32
      aiex.npu.write32(%addr, %v) : i32, i32
    }
  }
}

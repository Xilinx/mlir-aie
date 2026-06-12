//===- scf_control_flow.mlir - SCF control flow in EmitC TXN ----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Characterization tests for C++ TXN generation through SCF control flow.
// The scf/arith ops are lowered by the upstream SCFToEmitC / ArithToEmitC
// conversions; only the AIEX txn ops become aie_runtime calls. Each check
// asserts the *intended* C++ behavior, not just incidental output.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-txn-cpp %s | FileCheck %s

// -----------------------------------------------------------------------------
// Nested scf.for: the inner txn_append_write32 must execute inside both loops,
// i.e. it is emitted within two nested `for` statements.
// -----------------------------------------------------------------------------
// CHECK-LABEL: generate_txn_nested_for
// CHECK: for (size_t [[I:.+]] = {{.*}}; [[I]] < {{.*}}; [[I]] +=
// CHECK:   for (size_t [[J:.+]] = {{.*}}; [[J]] < {{.*}}; [[J]] +=
// CHECK:     aie_runtime::txn_append_write32(txn,
// CHECK:     op_count++;

module {
  aie.device(npu2) {
    aie.runtime_sequence @nested_for(%buf : memref<16xi32>, %n : i32) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c0i = arith.index_cast %c0 : i32 to index
      %c1i = arith.index_cast %c1 : i32 to index
      %ni = arith.index_cast %n : i32 to index
      scf.for %i = %c0i to %ni step %c1i {
        scf.for %j = %c0i to %ni step %c1i {
          %iv = arith.index_cast %i : index to i32
          aiex.npu.write32(%iv, %n) : i32, i32
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// scf.if with a yielded result: the value written by the following write32 must
// be the value selected by the condition (100 when true, 200 when false), not
// the condition itself. Non-0/1 yields make the selection explicit.
// -----------------------------------------------------------------------------
// CHECK-LABEL: generate_txn_if_yield
// CHECK-DAG: int32_t [[T:.+]] = 100
// CHECK-DAG: int32_t [[F:.+]] = 200
// CHECK: int32_t [[SEL:v.+]];
// CHECK: if ({{.*}}) {
// CHECK:   [[SEL]] = [[T]];
// CHECK: } else {
// CHECK:   [[SEL]] = [[F]];
// CHECK: }
// CHECK: int32_t [[USE:v.+]] = [[SEL]];
// CHECK: uint32_t [[U:v.+]] = (uint32_t) [[USE]];
// CHECK: aie_runtime::txn_append_write32(txn, [[U]],

module {
  aie.device(npu2) {
    aie.runtime_sequence @if_yield(%buf : memref<16xi32>, %n : i32) {
      %c0 = arith.constant 0 : i32
      %c100 = arith.constant 100 : i32
      %c200 = arith.constant 200 : i32
      %cmp = arith.cmpi sgt, %n, %c0 : i32
      %sel = scf.if %cmp -> (i32) {
        scf.yield %c100 : i32
      } else {
        scf.yield %c200 : i32
      }
      aiex.npu.write32(%sel, %n) : i32, i32
    }
  }
}

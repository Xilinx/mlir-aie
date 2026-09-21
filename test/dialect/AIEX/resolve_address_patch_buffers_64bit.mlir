//===- resolve_address_patch_buffers_64bit.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Resolving a buffer adds its traced byte offset to arg_plus. aie-rt carries
// that field as u64, so the sum is computed at 64 bits and narrowed back only
// when it fits, and a runtime arg_plus is widened to match its addend.

// RUN: aie-opt %s -aie-resolve-address-patch-buffers --split-input-file | FileCheck %s

// A sum that fits keeps the i32 form.
// CHECK-LABEL: @fits
// CHECK: %[[C:.*]] = arith.constant 32 : i32
// CHECK: aiex.npu.address_patch(%[[C]] : i32) {addr = 119300 : ui32, arg_idx = 0 : i32}
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @fits(%arg0: memref<8xi32>) {
      %plus = arith.constant 16 : i32
      %view = memref.subview %arg0[4][4][1]
          : memref<8xi32> to memref<4xi32, strided<[1], offset: 4>>
      aiex.npu.address_patch(%plus : i32) buffer %view
          : memref<4xi32, strided<[1], offset: 4>> {addr = 119300 : ui32}
    }
  }
}

// -----

// A base already past 32 bits stays i64 across the addition.
// CHECK-LABEL: @wide
// CHECK: %[[C:.*]] = arith.constant 8589934608 : i64
// CHECK: aiex.npu.address_patch(%[[C]] : i64) {addr = 119300 : ui32, arg_idx = 0 : i32}
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @wide(%arg0: memref<8xi32>) {
      %plus = arith.constant 8589934592 : i64
      %view = memref.subview %arg0[4][4][1]
          : memref<8xi32> to memref<4xi32, strided<[1], offset: 4>>
      aiex.npu.address_patch(%plus : i64) buffer %view
          : memref<4xi32, strided<[1], offset: 4>> {addr = 119300 : ui32}
    }
  }
}

// -----

// A runtime i64 arg_plus takes an i64 addend, so the arith.addi is well-typed.
// CHECK-LABEL: @runtime_i64
// CHECK: %[[S:.*]] = arith.constant 16 : i64
// CHECK: %[[A:.*]] = arith.addi %{{.*}}, %[[S]] : i64
// CHECK: aiex.npu.address_patch(%[[A]] : i64) {addr = 119300 : ui32, arg_idx = 0 : i32}
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @runtime_i64(%arg0: memref<8xi32>, %plus: i64) {
      %view = memref.subview %arg0[4][4][1]
          : memref<8xi32> to memref<4xi32, strided<[1], offset: 4>>
      aiex.npu.address_patch(%plus : i64) buffer %view
          : memref<4xi32, strided<[1], offset: 4>> {addr = 119300 : ui32}
    }
  }
}

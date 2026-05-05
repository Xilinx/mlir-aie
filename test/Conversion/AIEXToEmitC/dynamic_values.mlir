//===- dynamic_values.mlir - Dynamic SSA values in EmitC TXN -----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests C++ TXN generation with dynamic SSA operands: runtime_sequence
// parameters flow through to dynamic npu.write32, npu.sync, and
// npu.address_patch operations. Verifies the generated C++ uses function
// parameters and (uint32_t) casts instead of only constants.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-txn-cpp %s | FileCheck %s

// The function should accept the i32 parameter (memref args are dropped).
// CHECK-LABEL: generate_txn_sequence
// CHECK-SAME: int32_t

// CHECK: std::vector<uint32_t> txn;
// CHECK: aie_runtime::txn_init(txn);
// CHECK: uint32_t op_count = 0;

// Dynamic write32: address and value come from SSA operands via uint32_t cast.
// CHECK: (uint32_t)
// CHECK: aie_runtime::txn_append_write32(txn,
// CHECK: op_count++;

// Dynamic sync: all parameters from SSA.
// CHECK: aie_runtime::txn_append_sync(txn,
// CHECK: op_count++;

// Dynamic address_patch: dyn_arg_plus from SSA.
// CHECK: aie_runtime::txn_append_address_patch(txn,
// CHECK: op_count++;

// CHECK: aie_runtime::txn_prepend_header(txn, op_count,
// CHECK: return txn;

module {
  aie.device(npu2) {
    aie.runtime_sequence(%buf : memref<16xi32>, %param : i32) {
      %c0_i32 = arith.constant 0 : i32

      // Dynamic write32 with SSA address and value
      aiex.npu.write32(%param, %param) {address = 0 : ui32, value = 0 : ui32} : i32, i32

      // Dynamic sync with SSA parameters
      aiex.npu.sync(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %param, %param) {channel = 0 : i32, column = 0 : i32, column_num = 0 : i32, direction = 0 : i32, row = 0 : i32, row_num = 0 : i32} : i32, i32, i32, i32, i32, i32

      // Dynamic address_patch with SSA arg_plus
      aiex.npu.address_patch(%param : i32) {addr = 100 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
    }
  }
}

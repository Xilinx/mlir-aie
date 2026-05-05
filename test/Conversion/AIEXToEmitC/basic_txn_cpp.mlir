//===- basic_txn_cpp.mlir - Basic EmitC TXN generation test ------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests the end-to-end C++ TXN generation pipeline (aie-translate
// --aie-generate-txn-cpp) with a runtime_sequence containing static
// npu.write32, npu.sync, and npu.address_patch operations.
//
// Verifies the generated C++ includes the expected function structure,
// TXN encoding calls, and op_count tracking.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-txn-cpp %s | FileCheck %s

// CHECK: #include "aie/Runtime/TxnEncoding.h"
// CHECK: #include <cstdint>
// CHECK: #include <vector>

// CHECK-LABEL: generate_txn_sequence
// CHECK: std::vector<uint32_t> txn;
// CHECK: aie_runtime::txn_init(txn);
// CHECK: uint32_t op_count = 0;

// CHECK: aie_runtime::txn_append_write32(txn,
// CHECK: op_count++;

// CHECK: aie_runtime::txn_append_sync(txn,
// CHECK: op_count++;

// CHECK: aie_runtime::txn_append_address_patch(txn,
// CHECK: op_count++;

// CHECK: aie_runtime::txn_prepend_header(txn, op_count,
// CHECK: return txn;

module {
  aie.device(npu2) {
    aie.runtime_sequence(%buf : memref<16xi32>) {
      aiex.npu.write32 {address = 196612 : ui32, value = 42 : ui32}
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      aiex.npu.address_patch {addr = 196616 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
    }
  }
}

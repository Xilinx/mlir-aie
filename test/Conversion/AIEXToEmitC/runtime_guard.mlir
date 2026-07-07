//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// The C++ TXN builder returns std::optional and an npu.assert_bd_field guard
// lowers to an early `return std::nullopt` on overflow: a runtime scalar that
// would overflow a narrow BD field yields no stream instead of a truncated one.

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s

// CHECK: #include <optional>
// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[P:v[0-9]+]]) {
// CHECK:   std::vector<uint32_t> txn;
// CHECK:   aie_runtime::txn_init(txn);
// CHECK:   if ([[P]] > 1023) return std::nullopt;
// CHECK:   aie_runtime::txn_append_write32(txn, {{v[0-9]+}}, [[P]]);
// CHECK:   return std::move(txn);
module {
  aie.device(npu1) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %n: i32) {
      %addr = arith.constant 119300 : i32
      aiex.npu.assert_bd_field(%n) {max = 1023 : i32} : i32
      aiex.npu.write32(%addr, %n) : i32, i32
    }
  }
}

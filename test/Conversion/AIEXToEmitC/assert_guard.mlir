//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A user-level shape constraint on a runtime scalar (cf.assert) lowers
// to an early `return std::nullopt` in the C++ TXN builder, so a dispatch with
// a violating value yields no stream instead of a wrong one.

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[K:v[0-9]+]]) {
// CHECK:   std::vector<uint32_t> txn;
// CHECK:   aie_runtime::txn_init(txn);
// CHECK:   if (!({{v[0-9]+}})) return aie_runtime::txn_refused("K must be a multiple of 64");
// CHECK:   aie_runtime::txn_append_write32(txn, {{v[0-9]+}}, [[K]]);
// CHECK:   return std::move(txn);
module {
  aie.device(npu1) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %k: i32) {
      %addr = arith.constant 119300 : i32
      %c64 = arith.constant 64 : i32
      %c0 = arith.constant 0 : i32
      %rem = arith.remsi %k, %c64 : i32
      %ok = arith.cmpi eq, %rem, %c0 : i32
      cf.assert %ok, "K must be a multiple of 64"
      aiex.npu.write32(%addr, %k) : i32, i32
    }
  }
}

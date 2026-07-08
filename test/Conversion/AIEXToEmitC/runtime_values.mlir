//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s -split-input-file --aie-npu-to-cpp | FileCheck %s

// The milestone's headline: runtime-sequence scalar arguments flow into the
// generated C++ as function parameters (not baked-in constants), so one xclbin
// can run at many values. Data fields (value/mask/sync counts/arg_plus) accept
// runtime values; the register address stays compile-time (it selects which
// hardware register, and is resolved with buffer/col/row folded in).

// CHECK: inline std::vector<uint32_t> generate_txn_main_seq(int32_t [[P:v[0-9]+]]) {
// CHECK:   std::vector<uint32_t> txn;
// CHECK:   aie_runtime::txn_init(txn);
// A runtime value flows straight into write32 as the parameter.
// CHECK:   aie_runtime::txn_append_write32(txn, {{v[0-9]+}}, [[P]]);
// Runtime sync counts come from the parameter.
// CHECK:   aie_runtime::txn_append_sync(txn, {{.*}}, [[P]], [[P]]);
// Runtime address_patch arg_plus comes from the parameter.
// CHECK:   aie_runtime::txn_append_address_patch(txn, {{.*}}, [[P]]);
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %param: i32) {
      %addr = arith.constant 119300 : i32
      %c0 = arith.constant 0 : i32
      aiex.npu.write32(%addr, %param) : i32, i32
      aiex.npu.sync(%c0, %c0, %c0, %c0, %param, %param) : i32, i32, i32, i32, i32, i32
      aiex.npu.address_patch(%param : i32) {addr = 119300 : ui32, arg_idx = 0 : i32}
    }
  }
}

// -----

// A runtime value derived through arith (param + 1) is lowered to emitc
// arithmetic by the convert-arith-to-emitc step, then fed to the txn call.
// CHECK: inline std::vector<uint32_t> generate_txn_main_derived(int32_t [[P:v[0-9]+]]) {
// CHECK:   {{v[0-9]+}} = {{v[0-9]+}} + {{v[0-9]+}};
// CHECK:   aie_runtime::txn_append_write32(txn,
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @derived(%arg0: memref<8xi32>, %param: i32) {
      %c1 = arith.constant 1 : i32
      %x = arith.addi %param, %c1 : i32
      %addr = arith.constant 100 : i32
      aiex.npu.write32(%addr, %x) : i32, i32
    }
  }
}

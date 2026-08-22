//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s -split-input-file --aie-npu-to-cpp | FileCheck %s

// npu.blockwrite_values carries a payload computed at TXN-build time, for the
// dynamic BD-pool path where both the register base and some payload words are
// runtime. It stages the words into a local C++ array and emits ONE
// txn_append_blockwrite over it -- as opposed to npu.blockwrite, whose payload
// is a compile-time memref.global inlined as an array initializer.

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[P:v[0-9]+]]) {
// CHECK:   uint32_t [[ARR:v[0-9]+]][3] = {};
// CHECK:   [[ARR]][0] = [[P]];
// CHECK:   [[ARR]][1] = {{v[0-9]+}};
// CHECK:   [[ARR]][2] = {{v[0-9]+}};
// CHECK:   aie_runtime::txn_append_blockwrite(txn, {{v[0-9]+}}, [[ARR]], {{v[0-9]+}},
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %param: i32) {
      %addr = arith.constant 119300 : i32
      %c7 = arith.constant 7 : i32
      %derived = arith.addi %param, %c7 : i32
      aiex.npu.blockwrite_values(%addr : i32) values %param, %c7, %derived : i32, i32, i32
    }
  }
}

// -----

// The shape the dynamic BD lowering actually produces: a RUNTIME register base
// (bdBase = 119300 + bd_id*32) and an address_patch landing inside that same
// block. The blockwrite must precede the patch, and both reference the runtime
// base -- this ordering is what an aiebu-style ELF consumer requires.

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_bd_pool(int32_t [[BD:v[0-9]+]]) {
// CHECK:   uint32_t [[ARR:v[0-9]+]][2] = {};
// CHECK:   aie_runtime::txn_append_blockwrite(txn, [[BASE:v[0-9]+]], [[ARR]],
// CHECK:   aie_runtime::txn_append_address_patch(txn,
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @bd_pool(%arg0: memref<8xi32>, %bd_id: i32) {
      %stride = arith.constant 32 : i32
      %base = arith.constant 119300 : i32
      %off = arith.muli %bd_id, %stride : i32
      %bdbase = arith.addi %base, %off : i32
      %len = arith.constant 256 : i32
      %valid = arith.constant 33554432 : i32
      aiex.npu.blockwrite_values(%bdbase : i32) values %len, %valid : i32, i32
      %c4 = arith.constant 4 : i32
      %patch_addr = arith.addi %bdbase, %c4 : i32
      %arg_plus = arith.constant 0 : i32
      aiex.npu.address_patch(%arg_plus : i32) addr %patch_addr : i32 {addr = 0 : ui32, arg_idx = 0 : i32}
    }
  }
}

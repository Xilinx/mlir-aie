//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s --check-prefix=FOLD
// RUN: aie-translate %s --aie-npu-to-cpp --aie-npu-fold-ddr-addr-offset=false | FileCheck %s --check-prefix=NOFOLD

// The fold rule itself lives in TxnEncoding.h's txn_append_arg_patch, which the
// static path calls too; this pass only decides the flag. =false (HRX,
// full-ELF) never folds.

// FOLD: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[P:v[0-9]+]]) {
// FOLD:   [[F:v[0-9]+]] = true
// FOLD:   aie_runtime::txn_append_arg_patch(txn, {{v[0-9]+}}, {{.*}}, [[P]], [[F]]);
// FOLD:   [[F2:v[0-9]+]] = true
// FOLD:   aie_runtime::txn_append_arg_patch(txn, {{v[0-9]+}}, {{.*}}, [[P]], [[F2]]);

// NOFOLD: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[P:v[0-9]+]]) {
// NOFOLD:   [[F:v[0-9]+]] = false
// NOFOLD:   aie_runtime::txn_append_arg_patch(txn, {{v[0-9]+}}, {{.*}}, [[P]], [[F]]);
// NOFOLD:   [[F2:v[0-9]+]] = false
// NOFOLD:   aie_runtime::txn_append_arg_patch(txn, {{v[0-9]+}}, {{.*}}, [[P]], [[F2]]);

// The generated header is built by field name, so reordering TxnDeviceInfo
// cannot silently mis-encode it.
// FOLD: aie_runtime::txn_prepend_header(txn, {{.*}}, aie_runtime::txn_device_info(3, 6, 1, 1));
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %param: i32) {
      aiex.npu.address_patch(%param : i32) {addr = 119300 : ui32, arg_idx = 2 : i32}
      aiex.npu.address_patch(%param : i32) {addr = 119300 : ui32, arg_idx = 6 : i32}
    }
  }
}

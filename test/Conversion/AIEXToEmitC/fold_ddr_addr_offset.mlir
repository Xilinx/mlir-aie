//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s --check-prefix=FOLD
// RUN: aie-translate %s --aie-npu-to-cpp --aie-npu-fold-ddr-addr-offset=false | FileCheck %s --check-prefix=NOFOLD

// The firmware auto-translates host buffer addresses for only the first 5
// arguments (arg_idx 0-4); a DDR address_patch for a later argument must fold
// the AIE DDR aperture offset (0x80000000) into arg_plus itself to land
// correctly -- mirroring what AIETargetNPU.cpp's binary emitter does for the
// static path. arg_idx < 5 is never folded either way; arg_idx >= 5 is folded
// only when --aie-npu-fold-ddr-addr-offset=true (the default, matching the
// xclbin + instruction-buffer runtime). --aie-npu-fold-ddr-addr-offset=false
// (HRX, full-ELF) must leave arg_plus untouched at every arg_idx.

// FOLD: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[P:v[0-9]+]]) {
// arg_idx=2 (< 5): untouched, passed straight through.
// FOLD:   aie_runtime::txn_append_address_patch(txn, {{v[0-9]+}}, {{.*}}, [[P]]);
// arg_idx=6 (>= 5): folded -- 0x80000000 added before the call.
// FOLD:   [[OFF:v[0-9]+]] = -2147483648
// FOLD:   {{v[0-9]+}} = (uint32_t) [[P]]
// FOLD:   {{v[0-9]+}} = (uint32_t) [[OFF]]
// FOLD:   [[SUM:v[0-9]+]] = {{v[0-9]+}} + {{v[0-9]+}}
// FOLD:   [[FOLDED:v[0-9]+]] = (int32_t) [[SUM]]
// FOLD:   aie_runtime::txn_append_address_patch(txn, {{v[0-9]+}}, {{.*}}, [[FOLDED]]);

// NOFOLD: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(int32_t [[P:v[0-9]+]]) {
// Neither arg_idx is folded: both address_patch calls use the raw parameter.
// NOFOLD:   aie_runtime::txn_append_address_patch(txn, {{v[0-9]+}}, {{.*}}, [[P]]);
// NOFOLD:   aie_runtime::txn_append_address_patch(txn, {{v[0-9]+}}, {{.*}}, [[P]]);
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>, %param: i32) {
      aiex.npu.address_patch(%param : i32) {addr = 119300 : ui32, arg_idx = 2 : i32}
      aiex.npu.address_patch(%param : i32) {addr = 119300 : ui32, arg_idx = 6 : i32}
    }
  }
}

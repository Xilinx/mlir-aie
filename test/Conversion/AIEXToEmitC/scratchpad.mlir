//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// create_scratchpad adds a trailing uint64_t device-address parameter;
// update_from_scratchpad adds none.

// RUN: aie-translate %s --aie-npu-to-cpp | FileCheck %s

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_main_seq(uint64_t [[ADDR:v[0-9]+]]) {
// CHECK:   aie_runtime::txn_append_create_scratchpad(txn, {{v[0-9]+}}, {{v[0-9]+}}, [[ADDR]]);
// CHECK:   aie_runtime::txn_append_update_reg(txn, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});
module {
  aie.device(npu1_1col) {
    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      aiex.npu.create_scratchpad {size = 8 : ui32}
      aiex.npu.update_from_scratchpad<mul> {state_table_idx = 1 : ui8, func_arg = 2 : ui32, address = 119300 : ui32}
    }
  }
}

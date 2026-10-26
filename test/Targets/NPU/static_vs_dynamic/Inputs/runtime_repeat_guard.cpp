// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include GEN_HDR

#include <cassert>
#include <cstdint>

int main() {
  for (int32_t repeat : {0, 1, 255}) {
    assert(generate_txn_main_mem_repeat(repeat));
    assert(generate_txn_main_shim_repeat(repeat));
  }
  // Each of these used to build a stream whose push ran 0 or 44 or 255
  // repeats instead of failing.
  for (int32_t repeat : {-1, 256, 300, INT32_MAX}) {
    assert(!generate_txn_main_mem_repeat(repeat));
    assert(!generate_txn_main_shim_repeat(repeat));
  }
}

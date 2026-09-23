// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include GEN_HDR

#include <cassert>
#include <cstdint>

int main() {
  assert(generate_txn_main_mem_bytes(0, 4));
  assert(generate_txn_main_mem_bytes(1568764, 4));
  assert(!generate_txn_main_mem_bytes(1568768, 4));
  assert(!generate_txn_main_mem_bytes(-4, 4));
  for (int32_t offset : {1, 2, 3})
    assert(!generate_txn_main_mem_bytes(offset, 4));
  for (int32_t len : {-4, -1, 0, 1, 2, 3, 5, 524288, INT32_MAX})
    assert(!generate_txn_main_mem_bytes(0, len));
  assert(generate_txn_main_mem_bytes(0, 524284));

  assert(generate_txn_main_core_halves(0, 2));
  assert(generate_txn_main_core_halves(30718, 2));
  assert(!generate_txn_main_core_halves(30720, 2));
  assert(!generate_txn_main_core_halves(1, 2));
  assert(!generate_txn_main_core_halves(-2, 2));
  for (int32_t len : {-2, 0, 1, 3, 32768})
    assert(!generate_txn_main_core_halves(0, len));
  assert(generate_txn_main_core_halves(0, 32766));

  assert(generate_txn_main_mem_words(0));
  // Multiplication by four must not wrap an invalid offset back to zero.
  assert(!generate_txn_main_mem_words(1073741824));
  assert(!generate_txn_main_mem_words(INT32_MAX));

  assert(generate_txn_main_stride(8, 8));
  assert(!generate_txn_main_stride(8, 0));
  assert(!generate_txn_main_stride(8, -1));
  assert(!generate_txn_main_stride(8, 131073));

  assert(generate_txn_main_shim_bytes(4));
  for (int32_t len : {-4, 0, 1, 2, 3, 5})
    assert(!generate_txn_main_shim_bytes(len));
}

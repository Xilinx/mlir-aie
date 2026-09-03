//===- fallback_alloc_error.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses %s

// Buffer "b" is pinned inside bank 1, so the whole-bank buffer "a" cannot have
// the bank it requested. The pass does not retry: basic sequential allocation
// ignores mem_bank and would place "a" in a bank the design never requested.
module @test2 {
  aie.device(npu1) {

    // expected-error @below {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    // expected-error @below {{'aie.buffer' op requires 16384 bytes in bank 1, but only 12288 of 16384 bytes are free there}}
    %buf0 = aie.buffer(%tile34) { sym_name = "a", mem_bank = 1 : i32 } : memref<4096xi32> // use the whole buffer
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 16416 : i32, aligned = false } : memref<1024xi32> // allocate on bank_id 1
  }
}


// -----

// Two explicit addresses that overlap. Reported once, by bank-aware, with no
// fallback: an error has already been emitted, and basic sequential allocation
// honours explicit addresses too, so it would only rediscover the same overlap
// and report it a second time under a different message.
module @test3 {
  aie.device(npu1) {

    // expected-error @below {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) { sym_name = "a", address = 16384 : i32, mem_bank = 1 : i32 } : memref<4096xi32> // use the whole buffer
  // expected-error @below {{'aie.buffer' op would override allocated address}}
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 16416 : i32, aligned = false } : memref<1024xi32> // allocate on bank_id 1
  }
}

// -----

// A `mem_bank` that disagrees with the `address` beside it. This must not fall
// back either: basic sequential allocation ignores mem_bank entirely, so it
// would "succeed" and emit a buffer whose recorded bank is not the one the
// address actually lands in -- after an error had already been printed.
module @test4 {
  aie.device(npu1) {

    // expected-error @below {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    // expected-error @below {{'aie.buffer' op mem_bank attribute is inconsistent with address attribute}}
    %buf0 = aie.buffer(%tile34) { sym_name = "a", address = 16384 : i32, mem_bank = 2 : i32 } : memref<16xi32> // 16384 is in bank 1
    %buf1 = aie.buffer(%tile34) { sym_name = "b" } : memref<16xi32>
  }
}

//===- fallback_alloc_error.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses %s

// buffer "a" allocate the whole 16kB on bank_id 1. Thus, when buffer "b" tried to allocated a piece on bank1, it failed.
// This trigger a fallback to basic sequential allocation.
module @test2 {
  aie.device(npu1) {

    // expected-warning @below {{Bank-aware allocation failed, trying basic sequential allocation.}}
    %tile34 = aie.tile(3, 4)
    // expected-error @below {{'aie.buffer' op would override existing mem_bank}}
    %buf0 = aie.buffer(%tile34) { sym_name = "a", mem_bank = 1 : i32 } : memref<4096xi32> // use the whole buffer
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 16416 : i32, aligned = false } : memref<1024xi32> // allocate on bank_id 1
  }
}


// -----

module @test3 {
  aie.device(npu1) {

    // expected-warning @below {{Bank-aware allocation failed, trying basic sequential allocation.}}
    // expected-error @below {{'aie.tile' op Basic sequential allocation also failed.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) { sym_name = "a", address = 16384 : i32, mem_bank = 1 : i32 } : memref<4096xi32> // use the whole buffer
  // expected-error @below {{'aie.buffer' op would override allocated address}}
  // expected-error @below {{'aie.buffer' op buffer '"b"' at address 0x4020 overlaps with '"a"' at address 0x4000 (size: 16384 bytes)}}
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 16416 : i32, aligned = false } : memref<1024xi32> // allocate on bank_id 1
  }
}

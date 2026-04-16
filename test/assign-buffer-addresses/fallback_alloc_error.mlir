//===- bank_aware_prealloc_error.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses %s

// buffer "a" allocate the whole 16kB on bank_id 1. Thus, when buffer "b" tried to allocated a piece on bank1, it failed.
// This trigger a fallback to basic sequential allocation.
module @test2 {
  aie.device(npu1) {

    // expected-warning @below {{Bank-aware allocation failed, trying basic sequential allocation.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) { sym_name = "a", mem_bank = 1 : i32 } : memref<4096xi32> // use the whole buffer
    // expected-error@+1 {{'aie.buffer' op would override allocated address}}    
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 16416 : i32 } : memref<1024xi32> // allocate on bank_id 1
  }
}

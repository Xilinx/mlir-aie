//===- bank_aware_prealloc_error.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses='alloc-scheme=bank-aware' %s

module @test0 {
  aie.device(npu1) {
    // expected-error@+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) : memref<200xi32>
    %buf1 = aie.buffer(%tile34) : memref<100xi32>
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 4096 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op would override existing mem_bank}}
    %buf3 = aie.buffer(%tile34) { sym_name = "c", address = 12288 : i32 } : memref<1024xi32>
    %buf4 = aie.buffer(%tile34) { sym_name = "d", address = 20000 : i32 } : memref<1024xi32>
    %buf5 = aie.buffer(%tile34) : memref<800xi32>
  }
}

// -----

module @test1 {
  aie.device(npu1) {
    // expected-error@+1 {{'aie.tile' op Bank-aware allocation failed.}}
    %tile34 = aie.tile(3, 4)
    %buf0 = aie.buffer(%tile34) { sym_name = "a", address = 0 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op would override allocated address}}
    %buf2 = aie.buffer(%tile34) { sym_name = "b", address = 1024 : i32 } : memref<1024xi32>
  }
}

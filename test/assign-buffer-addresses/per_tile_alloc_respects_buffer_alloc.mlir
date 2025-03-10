//===- per_tile_alloc_respects_buffer_alloc.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s

module @test {
  aie.device(xcvc1902) {
    
    //Note that the bank-aware allocator should issue two different types of errors with these buffer allocation addresses.
    //This test is making sure that the alloc-scheme=basic-sequential flag is respected.

    %t1 = aie.tile(0, 1)
    //CHECK: address = 0
    %buf0 = aie.buffer(%t1) { address = 0 : i32 } : memref<1024xi32>
    //CHECK: address = 1024
    %buf2 = aie.buffer(%t1) { address = 1024 : i32 } : memref<1024xi32>
    //CHECK: address = 12288
    %b3 = aie.buffer(%t1) { address = 12288 : i32 } : memref<1024xi32>
    //CHECK: address = 20000
    %b34 = aie.buffer(%t1) { address = 20000 : i32 } : memref<1024xi32>

    aie.core(%t1) {
      aie.end
    }
  }
}

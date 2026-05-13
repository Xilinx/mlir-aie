//===- bank_aware_alloc_simple.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s
// CHECK: %a = aie.buffer(%tile_3_3) {address = 16384 : i32, mem_bank = 2 : i32, sym_name = "a"} : memref<16xi8> 
// CHECK: %b = aie.buffer(%tile_3_3) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "b"} : memref<512xi32> 
// CHECK: %c = aie.buffer(%tile_3_3) {address = 8192 : i32, mem_bank = 1 : i32, sym_name = "c"} : memref<16xi16> 
// CHECK: %_anonymous0 = aie.buffer(%tile_4_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "_anonymous0"} : memref<500xi32> 
// CHECK: %a1 = aie.buffer(%tile_3_3) {address = 2048 : i32, mem_bank = 0 : i32, sym_name = "a1"} : memref<4xi8> 
// CHECK: %b2 = aie.buffer(%tile_3_3) {address = 2052 : i32, aligned = false, mem_bank = 0 : i32, sym_name = "b2"} : memref<12xi8> 
// CHECK: %b3 = aie.buffer(%tile_3_3) {address = 2112 : i32, mem_bank = 0 : i32, sym_name = "b3"} : memref<4xi8>
module @test {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
    %1 = aie.buffer(%0) { sym_name = "b" } : memref<512xi32>
    %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
    %3 = aie.tile(4, 4)
    %4 = aie.buffer(%3) : memref<500xi32>
    aie.core(%0) {
      aie.end
    }
    aie.core(%3) {
      aie.end
    }
  }
}

// -----

module @test_align{
  aie.device(npu2) {
    %0 = aie.tile(3, 3)
    %b1 = aie.buffer(%0) { address = 2048 : i32, sym_name = "a1"} : memref<4xi8>
    %b2 = aie.buffer(%0) { mem_bank = 0 : i32, sym_name = "b2", aligned=false} : memref<12xi8>
    %b3 = aie.buffer(%0) { mem_bank = 0 : i32, sym_name = "b3"} : memref<4xi8>
    aie.core(%0) {
      aie.end
    }{stack_size = 2048 : i32}
    
  }

}
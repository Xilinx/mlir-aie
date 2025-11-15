//===- aie-vector-to-pointer-loops.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-vector-to-pointer-loops %s | FileCheck %s

// Test vector-to-pointer loop transformations

// CHECK-LABEL: @test1
aie.device(xcvc1902) @test1 {
  %tile = aie.tile(1, 1)
  %buf = aie.buffer(%tile) : memref<1024xi32, 2 : i32>
  
  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: ptr.to_ptr
    // CHECK: scf.for
    // CHECK-SAME: -> (!ptr.ptr<#ptr.generic_space>)
    %result = scf.for %i = %c0 to %c16 step %c1 iter_args(%idx = %c0) -> (index) {
      // CHECK: ptr.get_metadata
      // CHECK: ptr.from_ptr
      // CHECK: vector.load
      // CHECK-SAME: [%c0]
      %vec = vector.load %buf[%idx] : memref<1024xi32, 2 : i32>, vector<16xi32>
      // CHECK: vector.store
      // CHECK-SAME: [%c0]
      vector.store %vec, %buf[%idx] : memref<1024xi32, 2 : i32>, vector<16xi32>
      // CHECK: ptr.ptr_add
      %next_idx = arith.addi %idx, %c1 : index
      // CHECK: scf.yield
      scf.yield %next_idx : index
    }
    aie.end
  }
}

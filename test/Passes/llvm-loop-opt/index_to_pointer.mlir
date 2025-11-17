//===- index_to_pointer.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-llvm-loop-opt %s | FileCheck %s

// Test index-to-pointer loop transformation
// This pass converts index-carried loops to pointer-carried loops for better performance

module attributes {llvm.target_triple = "aie2p"} {
  
  // CHECK-LABEL: llvm.func @simple_loop_index_to_ptr
  llvm.func @simple_loop_index_to_ptr() {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c4 = llvm.mlir.constant(4 : index) : i64
    %c64 = llvm.mlir.constant(64 : index) : i64
    
    // Base pointer (simulating a buffer)
    %base_ptr = llvm.mlir.zero : !llvm.ptr
    
    llvm.br ^bb1
    
  // CHECK: ^bb1:
  // CHECK: %[[INIT_PTR:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  // CHECK: llvm.br ^bb2(%{{.*}}, %[[INIT_PTR]] : i64, !llvm.ptr)
  ^bb1:
    llvm.br ^bb2(%c0, %c0 : i64, i64)
    
  // CHECK: ^bb2(%{{.*}}: i64, %[[PTR_ARG:.*]]: !llvm.ptr):
  ^bb2(%iter: i64, %idx: i64):
    %cmp = llvm.icmp "slt" %iter, %c4 : i64
    llvm.cond_br %cmp, ^bb3, ^bb4
    
  // CHECK: ^bb3:
  // CHECK: llvm.load %[[PTR_ARG]] {{.*}} : !llvm.ptr -> vector<64xi8>
  ^bb3:
    // Original pattern: getelementptr base[index] -> load -> add index
    // Should be transformed to: load ptr -> getelementptr ptr[stride]
    %ptr = llvm.getelementptr %base_ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %val = llvm.load %ptr {alignment = 1 : i64} : !llvm.ptr -> vector<64xi8>
    
    %next_iter = llvm.add %iter, %c1 : i64
    
    // CHECK: %[[NEXT_PTR:.*]] = llvm.getelementptr %[[PTR_ARG]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %next_idx = llvm.add %idx, %c64 : i64
    
    // CHECK: llvm.br ^bb2(%{{.*}}, %[[NEXT_PTR]] : i64, !llvm.ptr)
    llvm.br ^bb2(%next_iter, %next_idx : i64, i64)
    
  ^bb4:
    llvm.return
  }
}

//===- nested_loops.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-llvm-loop-opt %s | FileCheck %s

// Test nested loop optimization with patterns from real workloads
// Shows which loops can be narrowed to i32 and which cannot

module attributes {llvm.target_triple = "aie2p"} {
  
  // CHECK-LABEL: llvm.func @nested_loops_with_multiplication
  llvm.func @nested_loops_with_multiplication(%base: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c8 = llvm.mlir.constant(8 : index) : i64
    %c16 = llvm.mlir.constant(16 : index) : i64
    %c64 = llvm.mlir.constant(64 : index) : i64
    %c1024 = llvm.mlir.constant(1024 : index) : i64
    %zero_vec = llvm.mlir.constant(dense<0> : vector<64xi32>) : vector<64xi32>
    
    llvm.br ^outer_loop(%c0 : i64)
    
  // Outer loop: has multiplication for address calculation
  // Currently CANNOT be narrowed (needs dataflow analysis)
  // CHECK: ^[[OUTER:bb[0-9]+]](%{{.*}}: i64):
  ^outer_loop(%i: i64):
    %cmp_i = llvm.icmp "slt" %i, %c8 : i64
    llvm.cond_br %cmp_i, ^middle_loop(%c0 : i64), ^exit
    
  ^middle_loop(%j: i64):
    %cmp_j = llvm.icmp "slt" %j, %c16 : i64
    llvm.cond_br %cmp_j, ^middle_body, ^outer_increment
    
  ^middle_body:
    // Address calculation: uses multiplication
    %offset_i = llvm.mul %i, %c1024 overflow<nsw> : i64
    %offset_j = llvm.mul %j, %c64 overflow<nsw> : i64
    %combined_offset = llvm.add %offset_i, %offset_j : i64
    
    llvm.br ^inner_loop(%c0, %zero_vec, %combined_offset : i64, vector<64xi32>, i64)
    
  // Inner loop: pure induction variable (no multiplication)
  // CAN be narrowed to i32!
  // CHECK: ^[[INNER:bb[0-9]+]](%{{.*}}: i64, %{{.*}}: vector<32xi64>, %{{.*}}: !llvm.ptr):
  ^inner_loop(%k: i64, %acc: vector<64xi32>, %idx: i64):
    %cmp_k = llvm.icmp "slt" %k, %c16 : i64
    llvm.cond_br %cmp_k, ^inner_body, ^middle_increment
    
  ^inner_body:
    %ptr = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %data = llvm.load %ptr {alignment = 1 : i64} : !llvm.ptr -> vector<64xi8>
    %next_idx = llvm.add %idx, %c64 : i64
    
    // Bitcast fusion pattern
    %acc_native = llvm.bitcast %acc : vector<64xi32> to vector<32xi64>
    %input = llvm.bitcast %data : vector<64xi8> to vector<16xi32>
    %const_vec = llvm.mlir.constant(dense<1> : vector<32xi16>) : vector<32xi16>
    %const_conf = llvm.mlir.constant(8 : i32) : i32
    
    %result_native = "xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf"(
        %input, %const_vec, %acc_native, %const_conf) 
        : (vector<16xi32>, vector<32xi16>, vector<32xi64>, i32) -> vector<32xi64>
    %result = llvm.bitcast %result_native : vector<32xi64> to vector<64xi32>
    
    %next_k = llvm.add %k, %c1 : i64
    llvm.br ^inner_loop(%next_k, %result, %next_idx : i64, vector<64xi32>, i64)
    
  ^middle_increment:
    %next_j = llvm.add %j, %c1 : i64
    llvm.br ^middle_loop(%next_j : i64)
    
  ^outer_increment:
    %next_i = llvm.add %i, %c1 : i64
    llvm.br ^outer_loop(%next_i : i64)
    
  ^exit:
    llvm.return
  }
  
  // Test simple loop with only icmp + add (should be narrowed)
  // CHECK-LABEL: llvm.func @simple_loop_narrowable
  llvm.func @simple_loop_narrowable() {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c100 = llvm.mlir.constant(100 : index) : i64
    
    llvm.br ^loop(%c0 : i64)
    
  // Should be narrowed to i32
  // CHECK: ^[[LOOP:bb[0-9]+]](%{{.*}}: i64):
  ^loop(%i: i64):
    %cmp = llvm.icmp "slt" %i, %c100 : i64
    llvm.cond_br %cmp, ^body, ^exit
    
  ^body:
    %next = llvm.add %i, %c1 : i64
    llvm.br ^loop(%next : i64)
    
  ^exit:
    llvm.return
  }
  
  // Test loop with multiplication (should NOT be narrowed)
  // CHECK-LABEL: llvm.func @loop_with_mul_not_narrowable
  llvm.func @loop_with_mul_not_narrowable() {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c10 = llvm.mlir.constant(10 : index) : i64
    %c64 = llvm.mlir.constant(64 : index) : i64
    
    llvm.br ^loop(%c0 : i64)
    
  // Should remain i64 (has multiplication)
  // CHECK: ^[[LOOP:bb[0-9]+]](%{{.*}}: i64):
  ^loop(%i: i64):
    %cmp = llvm.icmp "slt" %i, %c10 : i64
    llvm.cond_br %cmp, ^body, ^exit
    
  ^body:
    // Multiplication prevents narrowing
    %offset = llvm.mul %i, %c64 overflow<nsw> : i64
    %next = llvm.add %i, %c1 : i64
    llvm.br ^loop(%next : i64)
    
  ^exit:
    llvm.return
  }
}

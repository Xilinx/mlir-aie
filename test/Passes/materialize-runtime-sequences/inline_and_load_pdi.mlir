//===- inline_and_load_pdi.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// Test that aiex.configure blocks are properly transformed into load_pdi operations
// and that aiex.run calls are inlined

module {
  // CHECK-LABEL: aie.device(npu2) {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)
    
    // CHECK-DAG: %[[IA0:.+]] = arith.constant 100 : i32
    // CHECK-DAG: %[[IV0:.+]] = arith.constant 42 : i32
    // CHECK-DAG: %[[IA1:.+]] = arith.constant 200 : i32
    // CHECK-DAG: %[[IV1:.+]] = arith.constant 99 : i32
    // CHECK-LABEL: aie.runtime_sequence @main_seq
    // CHECK-SAME: (%[[ARG0:.*]]: memref<16xi32>)
    aie.runtime_sequence @main_seq(%arg0: memref<16xi32>) {
      // CHECK-NOT: aiex.configure
      // CHECK-NOT: aiex.run
      // CHECK: aiex.npu.load_pdi {device_ref = @config_a}
      // CHECK: aiex.npu.write32(%[[IA0]], %[[IV0]]) {column = 1 : i32, row = 0 : i32}
      // CHECK: aiex.npu.write32(%[[IA1]], %[[IV1]]) {column = 0 : i32, row = 0 : i32}
      aiex.configure @config_a {
        aiex.run @seq_a(%arg0) : (memref<16xi32>)
      }
      %w32_addr = arith.constant 200 : i32
      %w32_val = arith.constant 99 : i32
      aiex.npu.write32(%w32_addr, %w32_val) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
  
  // CHECK: aie.device(npu2) @config_a
  aie.device(npu2) @config_a {
    %tile10 = aie.tile(1, 0)
    
    aie.runtime_sequence @seq_a(%arg0: memref<16xi32>) {
      %w32_addr_1 = arith.constant 100 : i32
      %w32_val_1 = arith.constant 42 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) {column = 1 : i32, row = 0 : i32} : i32, i32
    }
  }
}

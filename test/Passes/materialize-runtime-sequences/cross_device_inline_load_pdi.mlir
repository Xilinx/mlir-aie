//===- cross_device_inline_load_pdi.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --split-input-file %s | FileCheck %s

// Test cross-device inlining of runtime sequences:
// 1. Inlined load_pdi operations are preserved with original device_ref
// 2. InsertLoadPdiForConfigurePattern adds load_pdi at start only if needed
// 3. When callee starts with load_pdi, no duplicate is added

//===----------------------------------------------------------------------===//
// TEST 1: Callee does NOT start with load_pdi - one should be added at start
//===----------------------------------------------------------------------===//

module {
  // The outer/caller device - anonymous (no symbol name)
  // CHECK: aie.device(npu2) {
  aie.device(npu2) {
    %tile00 = aie.tile(0, 0)
    
      // After inlining, we should have:
      // 1. A load_pdi added by InsertLoadPdiForConfigurePattern (since the first inlined op is write32, not load_pdi)
      // 2. All operations from the callee sequence inlined
      // 3. The inlined load_pdi operations preserved with their original device_ref
      //
      // The write32 address/value operands are arith.constants hoisted to the
      // enclosing device scope; capture them, then check the inlined sequence.
      // CHECK-DAG: %[[CA0:.+]] = arith.constant 100 : i32
      // CHECK-DAG: %[[CA1:.+]] = arith.constant 200 : i32
      // CHECK-DAG: %[[CA2:.+]] = arith.constant 300 : i32
      // CHECK: aie.runtime_sequence @caller_seq
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device}
      // CHECK: aiex.npu.write32(%[[CA0]],
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device}
      // CHECK: aiex.npu.write32(%[[CA1]],
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device}
      // CHECK: aiex.npu.write32(%[[CA2]],
      aie.runtime_sequence @caller_seq(%arg0: memref<16xi32>) {
      aiex.configure @callee_device {
        aiex.run @callee_seq(%arg0) : (memref<16xi32>)
      }
    }
  }
  
  // The callee device that contains the runtime sequence to be inlined
  // CHECK: aie.device(npu2) @callee_device
  aie.device(npu2) @callee_device {
    %tile10 = aie.tile(1, 0)
    
    // This sequence has load_pdi operations embedded in it for reconfiguration
    // between iterations. When inlined, these should be preserved.
    aie.runtime_sequence @callee_seq(%arg0: memref<16xi32>) {
      // First iteration
      %w32_addr = arith.constant 100 : i32
      %w32_val = arith.constant 1 : i32
      aiex.npu.write32(%w32_addr, %w32_val) {column = 0 : i32, row = 0 : i32} : i32, i32
      
      // Reconfigure for second iteration
      aiex.npu.load_pdi {device_ref = @callee_device}
      %w32_addr_1 = arith.constant 200 : i32
      %w32_val_1 = arith.constant 2 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) {column = 0 : i32, row = 0 : i32} : i32, i32
      
      // Reconfigure for third iteration
      aiex.npu.load_pdi {device_ref = @callee_device}
      %w32_addr_2 = arith.constant 300 : i32
      %w32_val_2 = arith.constant 3 : i32
      aiex.npu.write32(%w32_addr_2, %w32_val_2) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// TEST 2: Callee STARTS with load_pdi - no duplicate should be added
//===----------------------------------------------------------------------===//

module {
  // CHECK: aie.device(npu2) {
  aie.device(npu2) {
    %tile00 = aie.tile(0, 0)
    
      // After inlining, the callee's load_pdi is at the start of the configure block.
      // InsertLoadPdiForConfigurePattern should detect this and NOT add another one.
      // We should see exactly 3 load_pdi operations (from the callee), not 4.
      // CHECK-DAG: %[[DA0:.+]] = arith.constant 100 : i32
      // CHECK-DAG: %[[DA1:.+]] = arith.constant 200 : i32
      // CHECK-DAG: %[[DA2:.+]] = arith.constant 300 : i32
      // CHECK: aie.runtime_sequence @caller_seq2
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device2}
      // CHECK: aiex.npu.write32(%[[DA0]],
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device2}
      // CHECK: aiex.npu.write32(%[[DA1]],
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device2}
      // CHECK: aiex.npu.write32(%[[DA2]],
      aie.runtime_sequence @caller_seq2(%arg0: memref<16xi32>) {
      aiex.configure @callee_device2 {
        aiex.run @callee_seq2(%arg0) : (memref<16xi32>)
      }
    }
  }
  
  // The callee device - its runtime sequence starts with load_pdi
  // CHECK: aie.device(npu2) @callee_device2
  aie.device(npu2) @callee_device2 {
    %tile10 = aie.tile(1, 0)
    
    // This sequence STARTS with a load_pdi operation.
    // When inlined into a configure block, InsertLoadPdiForConfigurePattern
    // should NOT add another load_pdi at the start.
    aie.runtime_sequence @callee_seq2(%arg0: memref<16xi32>) {
      // First load_pdi at the very start
      aiex.npu.load_pdi {device_ref = @callee_device2}
      %w32_addr_3 = arith.constant 100 : i32
      %w32_val_3 = arith.constant 1 : i32
      aiex.npu.write32(%w32_addr_3, %w32_val_3) {column = 0 : i32, row = 0 : i32} : i32, i32
      
      // Second iteration
      aiex.npu.load_pdi {device_ref = @callee_device2}
      %w32_addr_4 = arith.constant 200 : i32
      %w32_val_4 = arith.constant 2 : i32
      aiex.npu.write32(%w32_addr_4, %w32_val_4) {column = 0 : i32, row = 0 : i32} : i32, i32
      
      // Third iteration
      aiex.npu.load_pdi {device_ref = @callee_device2}
      %w32_addr_5 = arith.constant 300 : i32
      %w32_val_5 = arith.constant 3 : i32
      aiex.npu.write32(%w32_addr_5, %w32_val_5) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

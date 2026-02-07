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
    
    // CHECK: aie.runtime_sequence @caller_seq
    aie.runtime_sequence @caller_seq(%arg0: memref<16xi32>) {
      // After inlining, we should have:
      // 1. A load_pdi added by InsertLoadPdiForConfigurePattern (since the first inlined op is write32, not load_pdi)
      // 2. All operations from the callee sequence inlined
      // 3. The inlined load_pdi operations preserved with their original device_ref
      
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device}
      // CHECK-NEXT: aiex.npu.write32 {address = 100 : ui32
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device}
      // CHECK-NEXT: aiex.npu.write32 {address = 200 : ui32
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device}
      // CHECK-NEXT: aiex.npu.write32 {address = 300 : ui32
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
      aiex.npu.write32 {address = 100 : ui32, column = 0 : i32, row = 0 : i32, value = 1 : ui32}
      
      // Reconfigure for second iteration
      aiex.npu.load_pdi {device_ref = @callee_device}
      aiex.npu.write32 {address = 200 : ui32, column = 0 : i32, row = 0 : i32, value = 2 : ui32}
      
      // Reconfigure for third iteration
      aiex.npu.load_pdi {device_ref = @callee_device}
      aiex.npu.write32 {address = 300 : ui32, column = 0 : i32, row = 0 : i32, value = 3 : ui32}
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
    
    // CHECK: aie.runtime_sequence @caller_seq2
    aie.runtime_sequence @caller_seq2(%arg0: memref<16xi32>) {
      // After inlining, the callee's load_pdi is at the start of the configure block.
      // InsertLoadPdiForConfigurePattern should detect this and NOT add another one.
      // We should see exactly 3 load_pdi operations (from the callee), not 4.
      
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device2}
      // CHECK-NEXT: aiex.npu.write32 {address = 100 : ui32
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device2}
      // CHECK-NEXT: aiex.npu.write32 {address = 200 : ui32
      // CHECK: aiex.npu.load_pdi {device_ref = @callee_device2}
      // CHECK-NEXT: aiex.npu.write32 {address = 300 : ui32
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
      aiex.npu.write32 {address = 100 : ui32, column = 0 : i32, row = 0 : i32, value = 1 : ui32}
      
      // Second iteration
      aiex.npu.load_pdi {device_ref = @callee_device2}
      aiex.npu.write32 {address = 200 : ui32, column = 0 : i32, row = 0 : i32, value = 2 : ui32}
      
      // Third iteration
      aiex.npu.load_pdi {device_ref = @callee_device2}
      aiex.npu.write32 {address = 300 : ui32, column = 0 : i32, row = 0 : i32, value = 3 : ui32}
    }
  }
}

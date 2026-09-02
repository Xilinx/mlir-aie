//===- cross_device_inline_load_pdi.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences --convert-aie-to-transaction --aie-npu-to-cert --split-input-file %s | FileCheck %s

// Test cross-device inlining of runtime sequences:
// 1. Inlined load_pdi operations are preserved with original device_ref
// 2. InsertLoadPdiForConfigurePattern adds load_pdi at start only if needed
// 3. When callee starts with load_pdi, no duplicate is added
// 4. Absorbed/callee devices are removed from output (only main device emitted)

//===----------------------------------------------------------------------===//
// TEST 1: Callee does NOT start with load_pdi - one should be added at start
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module {
module {
  // The outer/caller device - anonymous (no symbol name)
  // CHECK-NEXT: aie.device(npu2) {
  aie.device(npu2) {
    %tile00 = aie.tile(0, 0)

    aie.runtime_sequence @caller_seq(%arg0: memref<16xi32>) {
      // After inlining, we should have:
      // 1. A load_pdi added by InsertLoadPdiForConfigurePattern (since the first inlined op is write32, not load_pdi)
      // 2. All operations from the callee sequence inlined
      // 3. The inlined load_pdi operations preserved with their original device_ref

      aiex.configure @callee_device {
        aiex.run @callee_seq(%arg0) : (memref<16xi32>)
      }
    }
  }

  // The callee device that contains the runtime sequence to be inlined
  // This device will be absorbed and removed from the output
  aie.device(npu2) @callee_device {
    %tile10 = aie.tile(1, 0)

    // This sequence has load_pdi operations embedded in it for reconfiguration
    // between iterations. When inlined, these should be preserved.
    aie.runtime_sequence @callee_seq(%arg0: memref<16xi32>) {
      // First iteration
      %wa0 = arith.constant 100 : i32
      %wv0 = arith.constant 1 : i32
      aiex.npu.write32(%wa0, %wv0) {column = 0 : i32, row = 0 : i32} : i32, i32

      // Reconfigure for second iteration
      aiex.npu.load_pdi {device_ref = @callee_device}
      %wa1 = arith.constant 200 : i32
      %wv1 = arith.constant 2 : i32
      aiex.npu.write32(%wa1, %wv1) {column = 0 : i32, row = 0 : i32} : i32, i32

      // Reconfigure for third iteration
      aiex.npu.load_pdi {device_ref = @callee_device}
      %wa2 = arith.constant 300 : i32
      %wv2 = arith.constant 3 : i32
      aiex.npu.write32(%wa2, %wv2) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

// After conversion to CERT (a bare cert.job, no enclosing page yet):
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.load_pdi(1, @callee_device)
// CHECK-NEXT: aiex.cert.write32(100, 1)
// CHECK-NEXT: aiex.cert.load_pdi(1, @callee_device)
// CHECK-NEXT: aiex.cert.write32(200, 2)
// CHECK-NEXT: aiex.cert.load_pdi(1, @callee_device)
// CHECK-NEXT: aiex.cert.write32(300, 3)
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.section @callee_device {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }

// -----

//===----------------------------------------------------------------------===//
// TEST 2: Callee STARTS with load_pdi - no duplicate should be added
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module {
module {
  // CHECK-NEXT: aie.device(npu2) {
  aie.device(npu2) {
    %tile00 = aie.tile(0, 0)

    aie.runtime_sequence @caller_seq2(%arg0: memref<16xi32>) {
      // After inlining, the callee's load_pdi is at the start of the configure block.
      // InsertLoadPdiForConfigurePattern should detect this and NOT add another one.
      // We should see exactly 3 load_pdi operations (from the callee), not 4.

      aiex.configure @callee_device2 {
        aiex.run @callee_seq2(%arg0) : (memref<16xi32>)
      }
    }
  }

  // The callee device - its runtime sequence starts with load_pdi
  // This device will be absorbed and removed from the output
  aie.device(npu2) @callee_device2 {
    %tile10 = aie.tile(1, 0)

    // This sequence STARTS with a load_pdi operation.
    // When inlined into a configure block, InsertLoadPdiForConfigurePattern
    // should NOT add another load_pdi at the start.
    aie.runtime_sequence @callee_seq2(%arg0: memref<16xi32>) {
      // First load_pdi at the very start
      aiex.npu.load_pdi {device_ref = @callee_device2}
      %wa3 = arith.constant 100 : i32
      %wv3 = arith.constant 1 : i32
      aiex.npu.write32(%wa3, %wv3) {column = 0 : i32, row = 0 : i32} : i32, i32

      // Second iteration
      aiex.npu.load_pdi {device_ref = @callee_device2}
      %wa4 = arith.constant 200 : i32
      %wv4 = arith.constant 2 : i32
      aiex.npu.write32(%wa4, %wv4) {column = 0 : i32, row = 0 : i32} : i32, i32

      // Third iteration
      aiex.npu.load_pdi {device_ref = @callee_device2}
      %wa5 = arith.constant 300 : i32
      %wv5 = arith.constant 3 : i32
      aiex.npu.write32(%wa5, %wv5) {column = 0 : i32, row = 0 : i32} : i32, i32
    }
  }
}

// After conversion to CERT (a bare cert.job, no enclosing page yet):
// CHECK: aiex.cert.job(2) {
// CHECK-NEXT: aiex.cert.load_pdi(1, @callee_device2)
// CHECK-NEXT: aiex.cert.write32(100, 1)
// CHECK-NEXT: aiex.cert.load_pdi(1, @callee_device2)
// CHECK-NEXT: aiex.cert.write32(200, 2)
// CHECK-NEXT: aiex.cert.load_pdi(1, @callee_device2)
// CHECK-NEXT: aiex.cert.write32(300, 3)
// CHECK-NEXT: }
// CHECK-NEXT: aiex.cert.section @callee_device2 {
// CHECK-NEXT: aiex.cert.page {
// CHECK-NEXT: aiex.cert.job(1) {
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }

//===- scratchpad_wait_parameters_lowering.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A core with aiex.wait_scratchpad_parameters acquires its parameter-sync lock
// where each wait is, inside its loop, and not once at the top of its body;
// the runtime sequence still writes the parameters and sets that lock. A read
// in a loop nested in the waiting one is in order. A core without a wait keeps
// the top-of-body acquire.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-scratchpad-parameters %s | FileCheck %s --implicit-check-not=aiex.wait_scratchpad_parameters

// CHECK-DAG: %[[T02:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[T03:.*]] = aie.tile(0, 3)
// CHECK-DAG: %[[L02:.*]] = aie.lock(%[[T02]]) {init = 0 : i32}
// CHECK-DAG: %[[L03:.*]] = aie.lock(%[[T03]]) {init = 0 : i32}

// CHECK: aie.core(%[[T02]]) {
// CHECK-NOT: aie.use_lock
// CHECK: scf.while
// CHECK: } do {
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[L02]], AcquireGreaterEqual, %[[ONE]])
// CHECK: memref.load
// CHECK: scf.for
// CHECK: memref.load
// CHECK: aie.end

// CHECK: aie.core(%[[T03]]) {
// CHECK-NEXT: %[[ONE3:.*]] = arith.constant 1 : i32
// CHECK-NEXT: aie.use_lock(%[[L03]], Acquire, %[[ONE3]])
// CHECK: aie.end

// CHECK: aie.runtime_sequence
// CHECK: aiex.npu.create_scratchpad
// CHECK-DAG: aiex.set_lock(%[[L02]], %{{.*}})
// CHECK-DAG: aiex.set_lock(%[[L03]], %{{.*}})
module {
  aiex.scratchpad_parameter @n : i32
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        aiex.wait_scratchpad_parameters
        %v = aiex.read_scratchpad_parameter @n : i32
        scf.for %j = %c0 to %c1 step %c1 {
          %w = aiex.read_scratchpad_parameter @n : i32
        }
        scf.yield
      }
      aie.end
    }
    aie.core(%t03) {
      %v = aiex.read_scratchpad_parameter @n : i32
      aie.end
    }
    aie.runtime_sequence(%arg0 : memref<16xi32>) {
    }
  }
}

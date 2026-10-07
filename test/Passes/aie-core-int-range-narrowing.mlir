//===- aie-core-int-range-narrowing.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-core-int-range-narrowing %s | FileCheck %s

// Loops in cores and device functions whose bounds fit in i32 get i32
// induction variables. Loops that might not fit keep `index`, and runtime
// sequences are not touched.

// CHECK-LABEL: aie.device(npu2)
aie.device(npu2) {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  // CHECK: func.func @helper(%[[M:.*]]: memref<1024xi32>)
  // CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
  // CHECK:   %[[IDX:.*]] = arith.index_castui %[[I]] : i32 to index
  // CHECK:   memref.store %{{.*}}, %[[M]][%[[IDX]]]
  func.func @helper(%m: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant 7 : i32
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %v, %m[%i] : memref<1024xi32>
    }
    return
  }

  // CHECK: aie.core
  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cmax = arith.constant 9223372036854775807 : index
    %cu32 = arith.constant 4294967295 : index
    %v = arith.constant 7 : i32
    // The INT64_MAX loop that `range_(sys.maxsize)` emits keeps `index`, but
    // the loop nested in it is narrowed.
    // CHECK: scf.for %{{.*}} = %{{.*}} to %c9223372036854775807 step %{{.*}} {
    // CHECK:   scf.for %[[J:.*]] = %c0_i32 to %c32_i32 step %c1_i32 : i32 {
    // CHECK:     %[[JIDX:.*]] = arith.index_castui %[[J]] : i32 to index
    // CHECK:     memref.load %{{.*}}[%[[JIDX]]]
    scf.for %iter = %c0 to %cmax step %c1 {
      scf.for %i = %c0 to %c32 step %c1 {
        %x = memref.load %buf[%i] : memref<1024xi32>
        %y = arith.addi %x, %v : i32
        memref.store %y, %buf[%i] : memref<1024xi32>
      }
      func.call @helper(%buf) : (memref<1024xi32>) -> ()
    }
    // 0xFFFFFFFF does not fit in a signed i32.
    // CHECK: scf.for %{{.*}} = %{{.*}} to %c4294967295 step %{{.*}} {
    scf.for %i = %c0 to %cu32 step %c1 {
      memref.store %v, %buf[%c0] : memref<1024xi32>
    }
    aie.end
  }

  // CHECK: aie.runtime_sequence
  // CHECK:   scf.for %[[R:.*]] = %c0 to %c4 step %c1 {
  // CHECK:     arith.addi %[[R]], %c1 : index
  aie.runtime_sequence(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %j = arith.addi %i, %c1 : index
    }
  }
}

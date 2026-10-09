//===- aie-core-forever-loops.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-core-forever-loops %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: aie-opt --aie-core-forever-loops='min-trip-count=4294967295' %s | FileCheck %s --check-prefixes=CHECK,U32

// Loops in cores and device functions that run at least `min-trip-count`
// times and never read their counter lose the counter. Loops that read it or
// run fewer times are kept, and runtime sequences are not touched.

// CHECK-LABEL: aie.device(npu2)
aie.device(npu2) {
  %tile = aie.tile(0, 2)
  %tile2 = aie.tile(0, 3)
  %buf = aie.buffer(%tile) : memref<1024xi32>
  %buf2 = aie.buffer(%tile2) : memref<1024xi32>

  // CHECK: func.func @helper(%[[M:.*]]: memref<1024xi32>)
  // CHECK-NOT: scf.for
  // CHECK: scf.while : () -> () {
  // CHECK:   %[[T:.*]] = arith.constant true
  // CHECK:   scf.condition(%[[T]])
  // CHECK: } do {
  // CHECK:   memref.store %{{.*}}, %[[M]][%{{.*}}]
  // CHECK:   scf.yield
  // CHECK: }
  func.func @helper(%m: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cmax = arith.constant 9223372036854775807 : index
    %v = arith.constant 7 : i32
    scf.for %i = %c0 to %cmax step %c1 {
      memref.store %v, %m[%c0] : memref<1024xi32>
    }
    return
  }

  // CHECK: aie.core(%{{.*}}) {
  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %cmax = arith.constant 9223372036854775807 : index
    %cmax_even = arith.constant 9223372036854775806 : index
    %c2pow55 = arith.constant 36028797018963968 : index
    %cu32 = arith.constant 4294967295 : index
    %v = arith.constant 7 : i32
    // The INT64_MAX loop that `range_(sys.maxsize)` emits loses its counter;
    // the loop nested in it is kept.
    // CHECK: scf.while : () -> () {
    // CHECK:   scf.condition
    // CHECK: } do {
    // CHECK:   scf.for %[[J:.*]] = %c0 to %c32 step %c1 {
    // CHECK:     memref.load %{{.*}}[%[[J]]]
    // CHECK:   func.call @helper
    // CHECK:   scf.yield
    // CHECK: }
    scf.for %iter = %c0 to %cmax step %c1 {
      scf.for %i = %c0 to %c32 step %c1 {
        %x = memref.load %buf[%i] : memref<1024xi32>
        %y = arith.addi %x, %v : i32
        memref.store %y, %buf[%i] : memref<1024xi32>
      }
      func.call @helper(%buf) : (memref<1024xi32>) -> ()
    }
    // So does the same loop after objectFifo unrolling by 2: the trip count
    // halves but stays above the default threshold of 2^56. Its epilogue
    // after the loop is kept.
    // CHECK: scf.while : () -> () {
    // CHECK: } do {
    // CHECK:   memref.store %{{.*}}, %{{.*}}[%c0]
    // CHECK:   memref.store %{{.*}}, %{{.*}}[%c1]
    // CHECK:   scf.yield
    // CHECK: }
    // CHECK: memref.store %{{.*}}, %{{.*}}[%c0]
    scf.for %iter = %c0 to %cmax_even step %c2 {
      memref.store %v, %buf[%c0] : memref<1024xi32>
      memref.store %v, %buf[%c1] : memref<1024xi32>
    }
    memref.store %v, %buf[%c0] : memref<1024xi32>
    // Below the threshold, the counter is kept.
    // DEFAULT: scf.for %{{.*}} = %c0 to %c36028797018963968 step %c1 {
    // U32: scf.while : () -> () {
    scf.for %iter = %c0 to %c2pow55 step %c1 {
      memref.store %v, %buf[%c0] : memref<1024xi32>
    }
    // A loop that reads its counter keeps it.
    // CHECK: scf.for %[[K:.*]] = %c0 to %c9223372036854775807 step %c1 {
    // CHECK:   arith.index_cast %[[K]]
    scf.for %k = %c0 to %cmax step %c1 {
      %k32 = arith.index_cast %k : index to i32
      memref.store %k32, %buf[%c0] : memref<1024xi32>
    }
    // 0xFFFFFFFF is only treated as forever when opted in.
    // DEFAULT: scf.for %{{.*}} = %c0 to %c4294967295 step %c1 {
    // U32: scf.while : () -> () {
    // U32-NOT: scf.for
    // U32: } do {
    scf.for %i = %c0 to %cu32 step %c1 {
      memref.store %v, %buf[%c0] : memref<1024xi32>
    }
    // A short loop keeps its counter even when it does not read it.
    // CHECK: scf.for %{{.*}} = %c0 to %c8 step %c1 {
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %v, %buf[%c1] : memref<1024xi32>
    }
    aie.end
  }

  // Loop-carried values, as dynamic objectFifo lowering produces, are kept.
  // CHECK: aie.core(%{{.*}}) {
  // CHECK:   %[[R:.*]]:2 = scf.while (%[[A:.*]] = %c0_i32, %[[B:.*]] = %c1_i32) : (i32, i32) -> (i32, i32) {
  // CHECK:     %[[T2:.*]] = arith.constant true
  // CHECK:     scf.condition(%[[T2]]) %[[A]], %[[B]] : i32, i32
  // CHECK:   } do {
  // CHECK:   ^bb0(%[[X:.*]]: i32, %[[Y:.*]]: i32):
  // CHECK:     memref.store %[[X]], %{{.*}}[%c0]
  // CHECK:     scf.yield %[[Y]], %[[X]] : i32, i32
  // CHECK:   }
  // CHECK:   memref.store %[[R]]#1, %{{.*}}[%c0]
  %core2 = aie.core(%tile2) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cmax = arith.constant 9223372036854775807 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %r:2 = scf.for %iter = %c0 to %cmax step %c1
        iter_args(%a = %c0_i32, %b = %c1_i32) -> (i32, i32) {
      memref.store %a, %buf2[%c0] : memref<1024xi32>
      scf.yield %b, %a : i32, i32
    }
    memref.store %r#1, %buf2[%c0] : memref<1024xi32>
    aie.end
  }

  // CHECK: aie.runtime_sequence
  // CHECK:   scf.for %{{.*}} = %c0 to %c9223372036854775807 step %c1 {
  aie.runtime_sequence(%arg0: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cmax = arith.constant 9223372036854775807 : index
    scf.for %i = %c0 to %cmax step %c1 {
    }
  }
}

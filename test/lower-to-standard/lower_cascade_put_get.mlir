//===- lower_cascade_put_get.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-localize-locks --aie-standard-lowering --split-input-file %s | FileCheck %s

// CHECK: %[[VECREAD:.*]] = vector.transfer_read %collapse_shape{{.*}}[%c0{{.*}}], %cst{{.*}} {in_bounds = [true]}
// CHECK: %[[BITCAST:.*]] = vector.bitcast %[[VECREAD]] : vector<32xbf16> to vector<16xi32>
// CHECK: call @llvm.aie2.mcd.write.vec(%[[BITCAST]], %c1{{.*}}) : (vector<16xi32>, i32) -> ()

module @example0 {
 aie.device(npu1) {

  %t33 = aie.tile(3, 3)

  %l33_0 = aie.lock(%t33, 0) {init = 1 : i32}
  %l33_1 = aie.lock(%t33, 1) {init = 0 : i32}

  %buf152 = aie.buffer(%t33) { sym_name = "a" } : memref<32x64xbf16, 2 : i32>

  aie.core(%t33) {
    aie.use_lock(%l33_0, AcquireGreaterEqual, 1)
    // code
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %subview = memref.subview %buf152[0, 0] [1, 32] [1, 1] : memref<32x64xbf16, 2 : i32> to memref<1x32xbf16, strided<[64, 1]>, 2 : i32>
    %collapse_shape_111 = memref.collapse_shape %subview [[0, 1]] : memref<1x32xbf16, strided<[64, 1]>, 2 : i32> into memref<32xbf16, strided<[1]>, 2 : i32>
    %4 = vector.transfer_read %collapse_shape_111[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1]>, 2 : i32>, vector<32xbf16>
    aie.put_cascade(%4 : vector<32xbf16>)
    aie.use_lock(%l33_1, Release, 1)
    aie.end
  }
 }
}

// -----

// CHECK: %[[VECREAD2:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %cst{{.*}} {in_bounds = [true]}
// CHECK: %[[BITCAST2:.*]] = vector.bitcast %[[VECREAD2]] : vector<32xbf16> to vector<16xi32>
// CHECK: call @llvm.aie2.mcd.write.vec(%[[BITCAST2]], %c1{{.*}}) : (vector<16xi32>, i32) -> ()

module @example1 {
 aie.device(npu1) {

  %t33 = aie.tile(3, 3)

  %l33_0 = aie.lock(%t33, 0) {init = 1 : i32}
  %l33_1 = aie.lock(%t33, 1) {init = 0 : i32}

  %buf = aie.buffer(%t33) { sym_name = "b" } : memref<32xbf16, 2 : i32>

  aie.core(%t33) {
    aie.use_lock(%l33_0, AcquireGreaterEqual, 1)
    // simpler code - no subview or collapse_shape
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %vec = vector.transfer_read %buf[%c0], %cst {in_bounds = [true]} : memref<32xbf16, 2 : i32>, vector<32xbf16>
    aie.put_cascade(%vec : vector<32xbf16>)
    aie.use_lock(%l33_1, Release, 1)
    aie.end
  }
 }
}

// -----

// CHECK: %[[GETCAST:.*]] = call @llvm.aie2.scd.read.vec(%c1{{.*}}) : (i32) -> vector<16xi32>
// CHECK: %[[BITCAST3:.*]] = vector.bitcast %[[GETCAST]] : vector<16xi32> to vector<32xbf16>
// CHECK: vector.transfer_write %[[BITCAST3]], %collapse_shape{{.*}}[%c0{{.*}}] {in_bounds = [true]}

module @example2 {
 aie.device(npu1) {

  %t33 = aie.tile(3, 3)

  %l33_0 = aie.lock(%t33, 0) {init = 1 : i32}
  %l33_1 = aie.lock(%t33, 1) {init = 0 : i32}

  %buf152 = aie.buffer(%t33) { sym_name = "c" } : memref<32x64xbf16, 2 : i32>

  aie.core(%t33) {
    aie.use_lock(%l33_0, AcquireGreaterEqual, 1)
    // code with get_cascade
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %subview = memref.subview %buf152[%c0, %c1] [1, 32] [1, 1] : memref<32x64xbf16, 2 : i32> to memref<1x32xbf16, strided<[64, 1], offset: ?>, 2 : i32>
    %4 = aie.get_cascade() : vector<32xbf16>
    %collapse_shape_111 = memref.collapse_shape %subview [[0, 1]] : memref<1x32xbf16, strided<[64, 1], offset: ?>, 2 : i32> into memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
    vector.transfer_write %4, %collapse_shape_111[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
    aie.use_lock(%l33_1, Release, 1)
    aie.end
  }
 }
}

// -----

// CHECK: %[[GETCAST2:.*]] = call @llvm.aie2.scd.read.vec(%c1{{.*}}) : (i32) -> vector<16xi32>
// CHECK: %[[BITCAST4:.*]] = vector.bitcast %[[GETCAST2]] : vector<16xi32> to vector<32xbf16>
// CHECK: vector.transfer_write %[[BITCAST4]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}

module @example3 {
 aie.device(npu1) {

  %t33 = aie.tile(3, 3)

  %l33_0 = aie.lock(%t33, 0) {init = 1 : i32}
  %l33_1 = aie.lock(%t33, 1) {init = 0 : i32}

  %buf = aie.buffer(%t33) { sym_name = "d" } : memref<32xbf16, 2 : i32>

  aie.core(%t33) {
    aie.use_lock(%l33_0, AcquireGreaterEqual, 1)
    // simpler code - no subview or collapse_shape, direct get_cascade
    %c0 = arith.constant 0 : index
    %vec = aie.get_cascade() : vector<32xbf16>
    vector.transfer_write %vec, %buf[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, 2 : i32>
    aie.use_lock(%l33_1, Release, 1)
    aie.end
  }
 }
}

// -----

// CHECK: %[[VECREAD3:.*]] = vector.transfer_read %{{.*}}[%c0{{.*}}], %c0{{.*}} {in_bounds = [true]}
// CHECK: call @llvm.aie2.mcd.write.vec(%[[VECREAD3]], %c1{{.*}}) : (vector<16xi32>, i32) -> ()

module @example4 {
 aie.device(npu1) {

  %t33 = aie.tile(3, 3)

  %l33_0 = aie.lock(%t33, 0) {init = 1 : i32}
  %l33_1 = aie.lock(%t33, 1) {init = 0 : i32}

  %buf = aie.buffer(%t33) { sym_name = "e" } : memref<16xi32, 2 : i32>

  aie.core(%t33) {
    aie.use_lock(%l33_0, AcquireGreaterEqual, 1)
    // put_cascade with vector<16xi32> - no bitcast needed
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %vec = vector.transfer_read %buf[%c0], %c0_i32 {in_bounds = [true]} : memref<16xi32, 2 : i32>, vector<16xi32>
    aie.put_cascade(%vec : vector<16xi32>)
    aie.use_lock(%l33_1, Release, 1)
    aie.end
  }
 }
}

// -----

// CHECK: %[[GETCAST3:.*]] = call @llvm.aie2.scd.read.vec(%c1{{.*}}) : (i32) -> vector<16xi32>
// CHECK: vector.transfer_write %[[GETCAST3]], %{{.*}}[%c0{{.*}}] {in_bounds = [true]}

module @example5 {
 aie.device(npu1) {

  %t33 = aie.tile(3, 3)

  %l33_0 = aie.lock(%t33, 0) {init = 1 : i32}
  %l33_1 = aie.lock(%t33, 1) {init = 0 : i32}

  %buf = aie.buffer(%t33) { sym_name = "f" } : memref<16xi32, 2 : i32>

  aie.core(%t33) {
    aie.use_lock(%l33_0, AcquireGreaterEqual, 1)
    // get_cascade with vector<16xi32> - no bitcast needed
    %c0 = arith.constant 0 : index
    %vec = aie.get_cascade() : vector<16xi32>
    vector.transfer_write %vec, %buf[%c0] {in_bounds = [true]} : vector<16xi32>, memref<16xi32, 2 : i32>
    aie.use_lock(%l33_1, Release, 1)
    aie.end
  }
 }
}

//===- aie-hoist-vector-transfer-pointers.mlir -----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -aie-hoist-vector-transfer-pointers -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @hoist_vector_transfer_read
func.func @hoist_vector_transfer_read(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c64 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %arg0[%i], %cst : memref<256xf32>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]]
    vector.transfer_write %v, %arg1[%i] : vector<16xf32>, memref<256xf32>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    // CHECK: arith.addi %[[PTR1]]
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}

// -----

// CHECK-LABEL: func.func @hoist_vector_transfer_write
func.func @hoist_vector_transfer_write(%arg0: memref<256xf32>, %arg1: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR:.*]] = %{{.*}})
  scf.for %i = %c0 to %c64 step %c1 {
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR]]] {in_bounds = [true]}
    vector.transfer_write %arg1, %arg0[%i] : vector<16xf32>, memref<256xf32>
    // CHECK: arith.addi %[[PTR]], %{{.*}}
    // CHECK: scf.yield %{{.*}}
  }
  return
}

// -----

// CHECK-LABEL: func.func @hoist_2d_memref
func.func @hoist_2d_memref(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %arg0[%i, %c0], %cst : memref<16x16xf32>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]], %{{.*}}
    vector.transfer_write %v, %arg1[%i, %c0] : vector<16xf32>, memref<16x16xf32>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    // CHECK: arith.addi %[[PTR1]], %{{.*}}
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}

// -----

// Test strided memref from subview - should preserve stride information in collapse_shape
// CHECK-LABEL: func.func @hoist_strided_memref
func.func @hoist_strided_memref(%arg0: memref<16x16x4x4xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant dense<0.0> : vector<1x1x4x4xf32>
  
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
    scf.for %j = %c0 to %c16 step %c1 {
      // CHECK: memref.subview %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1]
      // CHECK-SAME: memref<16x16x4x4xf32, 2> to memref<1x1x4x4xf32, strided<[256, 16, 4, 1], offset: ?>, 2>
      %subview = memref.subview %arg0[%i, %j, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] 
        : memref<16x16x4x4xf32, 2> to memref<1x1x4x4xf32, strided<[256, 16, 4, 1], offset: ?>, 2>
      // Subviews are created inside nested loops, so the pass should skip transforming them
      // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
      vector.transfer_write %cst, %subview[%c0, %c0, %c0, %c0] 
        {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf32>, memref<1x1x4x4xf32, strided<[256, 16, 4, 1], offset: ?>, 2>
    }
  }
  return
}

// -----

// Test collapse_shape preserves contiguous strided layout (offset:0 gets canonicalized)
// CHECK-LABEL: func.func @preserve_contiguous_strided_layout
func.func @preserve_contiguous_strided_layout(%arg0: memref<16x16xf32, 2>, %arg1: memref<16x16xf32, 2>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  
  // Create contiguous strided subviews outside loop
  %subview0 = memref.subview %arg0[0, 0] [16, 16] [1, 1] 
    : memref<16x16xf32, 2> to memref<16x16xf32, strided<[16, 1], offset: 0>, 2>
  %subview1 = memref.subview %arg1[0, 0] [16, 16] [1, 1] 
    : memref<16x16xf32, 2> to memref<16x16xf32, strided<[16, 1], offset: 0>, 2>
  
  // Offset of 0 gets canonicalized away in the output
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK-SAME: memref<16x16xf32, strided<[16, 1]>, 2> into memref<256xf32, strided<[1]>, 2>
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK-SAME: memref<16x16xf32, strided<[16, 1]>, 2> into memref<256xf32, strided<[1]>, 2>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %subview0[%i, %c0], %cst 
      : memref<16x16xf32, strided<[16, 1], offset: 0>, 2>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]], %{{.*}}
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    vector.transfer_write %v, %subview1[%i, %c0] 
      : vector<16xf32>, memref<16x16xf32, strided<[16, 1], offset: 0>, 2>
    // CHECK: arith.addi %[[PTR1]], %{{.*}}
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}

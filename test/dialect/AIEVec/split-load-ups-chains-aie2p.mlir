//===- split-load-ups-chains-aie2p.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --aievec-split-load-ups-chains | FileCheck %s

// This test validates the optimization that splits vector.load + aievec.ups chains
// for AIE2p targets to reduce the number of shuffle operations from 3 to 1.

module {
  // CHECK-LABEL: func.func @split_load_ups_v64i16_to_v64i32
  func.func @split_load_ups_v64i16_to_v64i32(%arg0: memref<128xi16>) -> vector<64xi32> {
    %c0 = arith.constant 0 : index
    
    // CHECK-NOT: vector.load{{.*}}vector<64xi16>
    // CHECK: [[LOAD0:%.+]] = vector.load %arg0[%c0] : memref<128xi16>, vector<32xi16>
    // CHECK: [[LOAD1:%.+]] = vector.load %arg0[%c32] : memref<128xi16>, vector<32xi16>
    // CHECK: [[UPS0:%.+]] = aievec.ups [[LOAD0]] {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
    // CHECK: [[UPS1:%.+]] = aievec.ups [[LOAD1]] {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
    // CHECK: [[RESULT:%.+]] = vector.shuffle [[UPS0]], [[UPS1]]
    // CHECK-SAME: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    // CHECK-SAME: : vector<32xi32>, vector<32xi32>
    // CHECK: return [[RESULT]] : vector<64xi32>
    
    %0 = vector.load %arg0[%c0] : memref<128xi16>, vector<64xi16>
    %1 = aievec.ups %0 {shift = 0 : i8} : vector<64xi16>, vector<64xi32>
    return %1 : vector<64xi32>
  }

  // CHECK-LABEL: func.func @no_split_smaller_vector
  // This should not be transformed (512-bit vector, not 1024-bit)
  func.func @no_split_smaller_vector(%arg0: memref<64xi16>) -> vector<32xi32> {
    %c0 = arith.constant 0 : index
    
    // CHECK: vector.load{{.*}}vector<32xi16>
    // CHECK: aievec.ups{{.*}}vector<32xi16>, vector<32xi32>
    // CHECK-NOT: vector.shuffle
    
    %0 = vector.load %arg0[%c0] : memref<64xi16>, vector<32xi16>
    %1 = aievec.ups %0 {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
    return %1 : vector<32xi32>
  }

  // CHECK-LABEL: func.func @no_split_multiple_uses
  // This should not be transformed (load has multiple uses)
  func.func @no_split_multiple_uses(%arg0: memref<128xi16>) -> (vector<64xi32>, vector<64xi16>) {
    %c0 = arith.constant 0 : index
    
    // CHECK: [[LOAD:%.+]] = vector.load{{.*}}vector<64xi16>
    // CHECK: [[UPS:%.+]] = aievec.ups [[LOAD]]{{.*}}vector<64xi16>, vector<64xi32>
    // CHECK-NOT: vector.shuffle
    // CHECK: return [[UPS]], [[LOAD]]
    
    %0 = vector.load %arg0[%c0] : memref<128xi16>, vector<64xi16>
    %1 = aievec.ups %0 {shift = 0 : i8} : vector<64xi16>, vector<64xi32>
    return %1, %0 : vector<64xi32>, vector<64xi16>
  }

  // CHECK-LABEL: func.func @split_with_offset
  func.func @split_with_offset(%arg0: memref<256xi16>) -> vector<64xi32> {
    %c64 = arith.constant 64 : index
    
    // CHECK-DAG: [[C64:%.+]] = arith.constant 64 : index
    // CHECK-DAG: [[C96:%.+]] = arith.constant 96 : index
    // CHECK: [[LOAD0:%.+]] = vector.load %arg0{{\[}}[[C64]]] : memref<256xi16>, vector<32xi16>
    // CHECK: [[LOAD1:%.+]] = vector.load %arg0{{\[}}[[C96]]] : memref<256xi16>, vector<32xi16>
    // CHECK: [[UPS0:%.+]] = aievec.ups [[LOAD0]]
    // CHECK: [[UPS1:%.+]] = aievec.ups [[LOAD1]]
    // CHECK: [[RESULT:%.+]] = vector.shuffle [[UPS0]], [[UPS1]]
    
    %0 = vector.load %arg0[%c64] : memref<256xi16>, vector<64xi16>
    %1 = aievec.ups %0 {shift = 0 : i8} : vector<64xi16>, vector<64xi32>
    return %1 : vector<64xi32>
  }

  // CHECK-LABEL: func.func @split_srs_store_v64i32_to_v64i16
  func.func @split_srs_store_v64i32_to_v64i16(%arg0: vector<64xi32>, %arg1: memref<128xi16>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    
    // CHECK-DAG: [[C32:%.+]] = arith.constant 32 : index
    // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
    // CHECK-DAG: [[C0_I32:%.+]] = arith.constant 0 : i32
    // CHECK: [[SHUFFLE0:%.+]] = vector.shuffle %arg0, %arg0
    // CHECK-SAME: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    // CHECK: [[SHUFFLE1:%.+]] = vector.shuffle %arg0, %arg0
    // CHECK-SAME: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    // CHECK: [[SRS0:%.+]] = aievec.srs [[SHUFFLE0]], [[C0_I32]] : vector<32xi32>, i32, vector<32xi16>
    // CHECK: [[SRS1:%.+]] = aievec.srs [[SHUFFLE1]], [[C0_I32]] : vector<32xi32>, i32, vector<32xi16>
    // CHECK: vector.store [[SRS0]], %arg1{{\[}}[[C0]]] : memref<128xi16>, vector<32xi16>
    // CHECK: vector.store [[SRS1]], %arg1{{\[}}[[C32]]] : memref<128xi16>, vector<32xi16>
    // CHECK-NOT: vector.store{{.*}}vector<64xi16>
    
    %0 = aievec.srs %arg0, %c0_i32 : vector<64xi32>, i32, vector<64xi16>
    vector.store %0, %arg1[%c0] : memref<128xi16>, vector<64xi16>
    return
  }

  // CHECK-LABEL: func.func @no_split_srs_store_smaller
  // This should not be transformed (1024-bit, not 2048-bit)
  func.func @no_split_srs_store_smaller(%arg0: vector<32xi32>, %arg1: memref<64xi16>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    
    // CHECK: [[SRS:%.+]] = aievec.srs %arg0{{.*}}vector<32xi32>, i32, vector<32xi16>
    // CHECK: vector.store [[SRS]]{{.*}}vector<32xi16>
    // CHECK-NOT: vector.shuffle
    
    %0 = aievec.srs %arg0, %c0_i32 : vector<32xi32>, i32, vector<32xi16>
    vector.store %0, %arg1[%c0] : memref<64xi16>, vector<32xi16>
    return
  }

  // CHECK-LABEL: func.func @no_split_srs_multiple_uses
  // This should not be transformed (SRS has multiple uses)
  func.func @no_split_srs_multiple_uses(%arg0: vector<64xi32>, %arg1: memref<128xi16>) -> vector<64xi16> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    
    // CHECK: [[SRS:%.+]] = aievec.srs %arg0{{.*}}vector<64xi32>, i32, vector<64xi16>
    // CHECK: vector.store [[SRS]]{{.*}}vector<64xi16>
    // CHECK-NOT: vector.shuffle
    // CHECK: return [[SRS]]
    
    %0 = aievec.srs %arg0, %c0_i32 : vector<64xi32>, i32, vector<64xi16>
    vector.store %0, %arg1[%c0] : memref<128xi16>, vector<64xi16>
    return %0 : vector<64xi16>
  }
}

//===- matmul-aie2p.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

func.func @matmul_aie2p_8x8x8(%A : vector<8x8xbf16>, %B : vector<8x8xbf16>,
                              %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16>
                                        into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK-LABEL: @matmul_aie2p_8x8x8
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>
// CHECK-DAG:  %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG:  %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG:  %[[FC:.*]] = vector.shape_cast %[[C]] : vector<8x8xf32> to vector<64xf32>
// Convert LHS to v64accfloat
// CHECK:      "xllvm.intr.aie2p.v32bf16.to.v32accfloat"
// CHECK:      "xllvm.intr.aie2p.v32bf16.to.v32accfloat"
// CHECK:      vector.shuffle {{.*}} : vector<32xf32>, vector<32xf32>
// Transpose RHS in bf16 format (more efficient than transposing in f32)
// CHECK:      llvm.bitcast %[[FB]] : vector<64xbf16> to vector<32xi32>
// CHECK:      vector.shuffle {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi32>, vector<32xi32>
// CHECK:      vector.shuffle {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi32>, vector<32xi32>
// CHECK:      llvm.mlir.constant(52 : i32) : i32
// CHECK:      llvm.mlir.constant(53 : i32) : i32
// CHECK:      "xllvm.intr.aie2p.vshuffle"({{.*}}, {{.*}}, {{.*}}) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK:      "xllvm.intr.aie2p.vshuffle"({{.*}}, {{.*}}, {{.*}}) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK:      vector.shuffle {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xi32>, vector<16xi32>
// CHECK:      llvm.bitcast {{.*}} : vector<32xi32> to vector<64xbf16>
// Convert transposed RHS to v64accfloat
// CHECK:      "xllvm.intr.aie2p.v32bf16.to.v32accfloat"
// CHECK:      "xllvm.intr.aie2p.v32bf16.to.v32accfloat"
// CHECK:      vector.shuffle {{.*}} : vector<32xf32>, vector<32xf32>
// Uses BFP16 format with conf=780
// CHECK:      llvm.mlir.constant(780 : i32) : i32
// CHECK:      llvm.bitcast %[[FC]] : vector<64xf32> to vector<64xi32>
// CHECK:      "xllvm.intr.aie2p.v64accfloat.to.v64bfp16ebs8"
// CHECK:      "xllvm.intr.aie2p.v64accfloat.to.v64bfp16ebs8"
// CHECK:      "xllvm.intr.aie2p.BFP576.BFP576.ACC2048.mac.conf"
// CHECK:      llvm.bitcast {{.*}} : vector<64xi32> to vector<64xf32>
// CHECK:      vector.shape_cast {{.*}} : vector<64xf32> to vector<8x8xf32>

func.func @matmul_aie2p_8x8x4(%A : vector<8x8xbf16>, %B : vector<8x4xbf16>,
                              %C : vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x8xbf16>, vector<8x4xbf16>
                                        into vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// CHECK-LABEL: @matmul_aie2p_8x8x4
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<8x4xf32>
// CHECK-DAG:  %[[FA:.*]] = vector.shape_cast %[[A]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG:  %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG:  %[[FC:.*]] = vector.shape_cast %[[C]] : vector<8x4xf32> to vector<32xf32>
// Uses shared helper perform8x8x4MatMul: split LHS into lower/upper 32, shuffle with modes 52/53
// CHECK:      vector.shuffle %[[FA]], %[[FA]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// CHECK:      vector.shuffle %[[FA]], %[[FA]] [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// CHECK:      "xllvm.intr.aie2p.vshuffle"({{.*}}, {{.*}}, {{.*}}) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// Performs 8 MAC operations with conf=60
// CHECK-COUNT-8: "xllvm.intr.aie2p.I512.I512.ACC1024.bf.mac.conf"
// CHECK:      vector.shape_cast {{.*}} : vector<32xf32> to vector<8x4xf32>

func.func @matmul_aie2p_4x8x8(%A : vector<4x8xbf16>, %B : vector<8x8xbf16>,
                              %C : vector<4x8xf32>) -> vector<4x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<4x8xbf16>, vector<8x8xbf16>
                                        into vector<4x8xf32>
  return %0 : vector<4x8xf32>
}

// CHECK-LABEL: @matmul_aie2p_4x8x8
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<4x8xf32>
// CHECK-DAG:  %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG:  %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x8xbf16> to vector<64xbf16>
// CHECK-DAG:  %[[FC:.*]] = vector.shape_cast %[[C]] : vector<4x8xf32> to vector<32xf32>
// CHECK:      llvm.mlir.constant(780 : i32) : i32
// Convert LHS v32bf16 to v32accfloat and pad to v64accfloat
// CHECK:      "xllvm.intr.aie2p.v32bf16.to.v32accfloat"(%[[FA]]) : (vector<32xbf16>) -> vector<32xf32>
// CHECK:      vector.shuffle {{.*}}, {{.*}} {{.*}}-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1{{.*}} : vector<32xf32>, vector<32xf32>
// Transpose RHS in bf16 format (more efficient than transposing in f32)
// CHECK:      llvm.bitcast %[[FB]] : vector<64xbf16> to vector<32xi32>
// CHECK:      vector.shuffle {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<32xi32>, vector<32xi32>
// CHECK:      vector.shuffle {{.*}} [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<32xi32>, vector<32xi32>
// CHECK:      llvm.mlir.constant(52 : i32) : i32
// CHECK:      llvm.mlir.constant(53 : i32) : i32
// CHECK:      "xllvm.intr.aie2p.vshuffle"({{.*}}, {{.*}}, {{.*}}) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK:      "xllvm.intr.aie2p.vshuffle"({{.*}}, {{.*}}, {{.*}}) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK:      vector.shuffle {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xi32>, vector<16xi32>
// CHECK:      llvm.bitcast {{.*}} : vector<32xi32> to vector<64xbf16>
// Convert transposed RHS to v64accfloat
// CHECK:      "xllvm.intr.aie2p.v32bf16.to.v32accfloat"
// CHECK:      "xllvm.intr.aie2p.v32bf16.to.v32accfloat"
// CHECK:      vector.shuffle {{.*}} : vector<32xf32>, vector<32xf32>
// Pad ACC and use BFP16 format
// CHECK:      llvm.bitcast %[[FC]] : vector<32xf32> to vector<32xi32>
// CHECK:      vector.shuffle {{.*}}, {{.*}} {{.*}}-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1{{.*}} : vector<32xi32>, vector<32xi32>
// CHECK:      "xllvm.intr.aie2p.v64accfloat.to.v64bfp16ebs8"
// CHECK:      "xllvm.intr.aie2p.v64accfloat.to.v64bfp16ebs8"
// CHECK:      "xllvm.intr.aie2p.BFP576.BFP576.ACC2048.mac.conf"
// Extract first 32 elements from result
// CHECK:      vector.shuffle {{.*}}, {{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xi32>, vector<64xi32>
// CHECK:      llvm.bitcast {{.*}} : vector<32xi32> to vector<32xf32>
// CHECK:      vector.shape_cast {{.*}} : vector<32xf32> to vector<4x8xf32>

func.func @matmul_aie2p_i512_acc2048_int(%A : vector<16xi32>, %B : vector<32xi16>,
                                         %C : vector<32xi64>) -> vector<32xi64> {
  %conf = arith.constant 0 : i32
  %0 = "xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf"(%A, %B, %C, %conf) :
      (vector<16xi32>, vector<32xi16>, vector<32xi64>, i32) -> vector<32xi64>
  return %0 : vector<32xi64>
}

// CHECK-LABEL: @matmul_aie2p_i512_acc2048_int
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<32xi16>, %[[C:.*]]: vector<32xi64>
// CHECK:      %[[CONF:.*]] = arith.constant 0 : i32
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf"(
// CHECK-SAME:         %[[A]], %[[B]], %[[C]], %[[CONF]]) :
// CHECK-SAME:         (vector<16xi32>, vector<32xi16>, vector<32xi64>, i32)
// CHECK-SAME:         -> vector<32xi64>
// CHECK:      return %[[RACC]] : vector<32xi64>

func.func @matmul_aie2p_i1024_acc2048_int(%A : vector<32xi32>, %B : vector<64xi16>,
                                          %C : vector<32xi64>) -> vector<32xi64> {
  %conf = arith.constant 0 : i32
  %0 = "xllvm.intr.aie2p.I1024.I1024.ACC2048.mac.conf"(%A, %B, %C, %conf) :
      (vector<32xi32>, vector<64xi16>, vector<32xi64>, i32) -> vector<32xi64>
  return %0 : vector<32xi64>
}

// CHECK-LABEL: @matmul_aie2p_i1024_acc2048_int
// CHECK-SAME: %[[A:.*]]: vector<32xi32>, %[[B:.*]]: vector<64xi16>, %[[C:.*]]: vector<32xi64>
// CHECK:      %[[CONF:.*]] = arith.constant 0 : i32
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2p.I1024.I1024.ACC2048.mac.conf"(
// CHECK-SAME:         %[[A]], %[[B]], %[[C]], %[[CONF]]) :
// CHECK-SAME:         (vector<32xi32>, vector<64xi16>, vector<32xi64>, i32)
// CHECK-SAME:         -> vector<32xi64>
// CHECK:      return %[[RACC]] : vector<32xi64>

func.func @matmul_aie2p_i512_acc2048(%A : vector<4x8xbf16>, %B : vector<8x4xbf16>,
                                     %C : vector<4x4xf32>) -> vector<4x4xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<4x8xbf16>, vector<8x4xbf16>
                                        into vector<4x4xf32>
  return %0 : vector<4x4xf32>
}

// CHECK-LABEL: @matmul_aie2p_i512_acc2048
// CHECK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x4xbf16>, %[[C:.*]]: vector<4x4xf32>
// CHECK-DAG:  %[[FA:.*]] = vector.shape_cast %[[A]] : vector<4x8xbf16> to vector<32xbf16>
// CHECK-DAG:  %[[FB:.*]] = vector.shape_cast %[[B]] : vector<8x4xbf16> to vector<32xbf16>
// CHECK-DAG:  %[[FC:.*]] = vector.shape_cast %[[C]] : vector<4x4xf32> to vector<16xf32>
// Pad LHS from 32 to 64 bfloat16
// CHECK:      vector.shuffle %[[FA]], %[[FA]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
// Pad ACC from 16 to 32 float
// CHECK:      vector.shuffle %[[FC]], %[[FC]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
// Uses shared helper perform8x8x4MatMul: split LHS, shuffle with modes 52/53, prepare 8 rows, prepare 8 columns
// CHECK:      "xllvm.intr.aie2p.vshuffle"({{.*}}, {{.*}}, {{.*}}) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// Performs 8 MAC operations
// CHECK-COUNT-8: "xllvm.intr.aie2p.I512.I512.ACC1024.bf.mac.conf"
// Extract first 16 elements from 32-element result
// CHECK:      vector.shuffle {{.*}} : vector<32xf32>, vector<32xf32>
// CHECK:      vector.shape_cast {{.*}} : vector<16xf32> to vector<4x4xf32>

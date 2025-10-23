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

func.func @matmul_aie2p(%A : vector<8x8xbf16>, %B : vector<8x8xbf16>,
                        %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x8xbf16>, vector<8x8xbf16>
                                        into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK-LABEL: @matmul_aie2p
// CHECK-SAME: %[[A:.*]]: vector<8x8xbf16>, %[[B:.*]]: vector<8x8xbf16>, %[[C:.*]]: vector<8x8xf32>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<8x8xbf16> to vector<64xbf16>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<8x8xbf16> to vector<64xbf16>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<8x8xf32> to vector<64xf32>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2p.I1024.I1024.ACC2048.bf.mac.conf"(
// CHECK-SAME:         %[[FA]], %[[FB]], %[[FC]], %[[CONF]]) :
// CHECK-SAME:         (vector<64xbf16>, vector<64xbf16>, vector<64xf32>, i32)
// CHECK-SAME:         -> vector<64xf32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[RACC]] :
// CHECK-SAME:                      vector<64xf32> to vector<8x8xf32>
// CHECK:      return %[[R]] : vector<8x8xf32>

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
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<4x8xbf16> to vector<32xbf16>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<8x4xbf16> to vector<32xbf16>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<4x4xf32> to vector<16xf32>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2p.I512.I512.ACC512.bf.mac.conf"(
// CHECK-SAME:         %[[FA]], %[[FB]], %[[FC]], %[[CONF]]) :
// CHECK-SAME:         (vector<32xbf16>, vector<32xbf16>, vector<16xf32>, i32)
// CHECK-SAME:         -> vector<16xf32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[RACC]] :
// CHECK-SAME:                      vector<16xf32> to vector<4x4xf32>
// CHECK:      return %[[R]] : vector<4x4xf32>

func.func @matmul_aie2p_i512_acc1024(%A : vector<8x4xbf16>, %B : vector<4x8xbf16>,
                                     %C : vector<8x8xf32>) -> vector<8x8xf32> {
  %0 = aievec.matmul_aie2p %A, %B, %C : vector<8x4xbf16>, vector<4x8xbf16>
                                        into vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK-LABEL: @matmul_aie2p_i512_acc1024
// CHECK-SAME: %[[A:.*]]: vector<8x4xbf16>, %[[B:.*]]: vector<4x8xbf16>, %[[C:.*]]: vector<8x8xf32>
// CHECK:      %[[FA:.*]] = vector.shape_cast %[[A]] :
// CHECK-SAME:                      vector<8x4xbf16> to vector<32xbf16>
// CHECK:      %[[FB:.*]] = vector.shape_cast %[[B]] :
// CHECK-SAME:                      vector<4x8xbf16> to vector<32xbf16>
// CHECK:      %[[FC:.*]] = vector.shape_cast %[[C]] :
// CHECK-SAME:                      vector<8x8xf32> to vector<64xf32>
// CHECK:      %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK:      %[[RACC:.*]] = "xllvm.intr.aie2p.I512.I512.ACC2048.bf.mac.conf"(
// CHECK-SAME:         %[[FA]], %[[FB]], %[[FC]], %[[CONF]]) :
// CHECK-SAME:         (vector<32xbf16>, vector<32xbf16>, vector<64xf32>, i32)
// CHECK-SAME:         -> vector<64xf32>
// CHECK:      %[[R:.*]] = vector.shape_cast %[[RACC]] :
// CHECK-SAME:                      vector<64xf32> to vector<8x8xf32>
// CHECK:      return %[[R]] : vector<8x8xf32>

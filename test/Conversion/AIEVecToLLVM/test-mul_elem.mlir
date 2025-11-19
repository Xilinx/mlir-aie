//===- test-mul_elem.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie2p" -split-input-file | FileCheck %s

// CHECK-LABEL: mul_elem_16_bf16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xbf16>
func.func @mul_elem_16_bf16(%lhs : vector<16xbf16>, %rhs : vector<16xbf16>) -> vector<16xf32> {
  // CHECK-DAG: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
  // CHECK-DAG: %[[PAD0:.*]] = vector.shuffle %[[LHS]], %[[LHS]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<16xbf16>, vector<16xbf16>
  // CHECK-DAG: %[[PAD1:.*]] = vector.shuffle %[[RHS]], %[[RHS]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<16xbf16>, vector<16xbf16>
  // CHECK: %[[MUL:.*]] = "xllvm.intr.aie2p.I512.I512.ACC512.bf.mul.conf"(%[[PAD0]], %[[PAD1]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, i32) -> vector<16xf32>
  %0 = aievec.mul_elem %lhs, %rhs : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// CHECK-LABEL: mul_elem_32_bf16
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<32xbf16>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<32xbf16>
func.func @mul_elem_32_bf16(%lhs : vector<32xbf16>, %rhs : vector<32xbf16>) -> vector<32xf32> {
  // CHECK: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
  // CHECK: %[[MUL:.*]] = "xllvm.intr.aie2p.I512.I512.ACC1024.bf.mul.conf"(%[[LHS]], %[[RHS]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, i32) -> vector<32xf32>
  %0 = aievec.mul_elem %lhs, %rhs : vector<32xbf16>, vector<32xbf16>, vector<32xf32>
  return %0 : vector<32xf32>
}

//===- test-bitwise-backend.mlir - bitwise vector legalization per backend -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// 512-bit integer vector arith.{andi,ori,xori} → aievec.{band,bor,bxor} only on
// the CPP (chess) backend; on the LLVMIR (Peano) backend the arith ops must
// pass through unchanged.

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=cpp" \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=NO-BITWISE-AIEVEC

// CPP-LABEL: func @vec_andi_512(
// CPP-SAME:    %[[LHS:.*]]: vector<16xi32>,
// CPP-SAME:    %[[RHS:.*]]: vector<16xi32>)
// LLVM-LABEL: func @vec_andi_512(
func.func @vec_andi_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CPP:  %[[R:.*]] = aievec.band %[[LHS]], %[[RHS]] : vector<16xi32>
  // CPP:  return %[[R]]
  // LLVM-NOT: aievec.band
  // LLVM:  %[[R:.*]] = arith.andi %{{.*}}, %{{.*}} : vector<16xi32>
  // LLVM:  return %[[R]]
  %r = arith.andi %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// CPP-LABEL: func @vec_ori_512(
// LLVM-LABEL: func @vec_ori_512(
func.func @vec_ori_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CPP:  aievec.bor %{{.*}}, %{{.*}} : vector<16xi32>
  // LLVM-NOT: aievec.bor
  // LLVM:  arith.ori %{{.*}}, %{{.*}} : vector<16xi32>
  %r = arith.ori %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// CPP-LABEL: func @vec_xori_512(
// LLVM-LABEL: func @vec_xori_512(
func.func @vec_xori_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // CPP:  aievec.bxor %{{.*}}, %{{.*}} : vector<16xi32>
  // LLVM-NOT: aievec.bxor
  // LLVM:  arith.xori %{{.*}}, %{{.*}} : vector<16xi32>
  %r = arith.xori %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// Non-512-bit vectors and scalars stay legal on both backends.

// CPP-LABEL: func @vec_andi_256(
// LLVM-LABEL: func @vec_andi_256(
func.func @vec_andi_256(%a: vector<8xi32>, %b: vector<8xi32>) -> vector<8xi32> {
  // CPP-NOT: aievec.band
  // LLVM-NOT: aievec.band
  // CPP:  arith.andi %{{.*}}, %{{.*}} : vector<8xi32>
  // LLVM: arith.andi %{{.*}}, %{{.*}} : vector<8xi32>
  %r = arith.andi %a, %b : vector<8xi32>
  return %r : vector<8xi32>
}

// CPP-LABEL: func @scalar_andi(
// LLVM-LABEL: func @scalar_andi(
func.func @scalar_andi(%a: i32, %b: i32) -> i32 {
  // CPP-NOT: aievec.band
  // LLVM-NOT: aievec.band
  // CPP:  arith.andi %{{.*}}, %{{.*}} : i32
  // LLVM: arith.andi %{{.*}}, %{{.*}} : i32
  %r = arith.andi %a, %b : i32
  return %r : i32
}

// NO-BITWISE-AIEVEC: module
// NO-BITWISE-AIEVEC-NOT: aievec.band
// NO-BITWISE-AIEVEC-NOT: aievec.bor
// NO-BITWISE-AIEVEC-NOT: aievec.bxor
// NO-BITWISE-AIEVEC-NOT: aievec.bneg

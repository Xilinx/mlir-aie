//===- test-bitwise-backend.mlir - bitwise vector legalization per backend -*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// 512-bit integer vector arith.{andi,ori,xori} → aievec.{band,bor,bxor} only on
// the CPP (chess) backend; on the LLVMIR (Peano) backend the arith ops must
// pass through unchanged.

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" \
// RUN:   | FileCheck %s --check-prefix=NO-BITWISE-AIEVEC

// LLVM-LABEL: func @vec_andi_512(
func.func @vec_andi_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // LLVM-NOT: aievec.band
  // LLVM:  %[[R:.*]] = arith.andi %{{.*}}, %{{.*}} : vector<16xi32>
  // LLVM:  return %[[R]]
  %r = arith.andi %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// LLVM-LABEL: func @vec_ori_512(
func.func @vec_ori_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // LLVM-NOT: aievec.bor
  // LLVM:  arith.ori %{{.*}}, %{{.*}} : vector<16xi32>
  %r = arith.ori %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// LLVM-LABEL: func @vec_xori_512(
func.func @vec_xori_512(%a: vector<16xi32>, %b: vector<16xi32>) -> vector<16xi32> {
  // LLVM-NOT: aievec.bxor
  // LLVM:  arith.xori %{{.*}}, %{{.*}} : vector<16xi32>
  %r = arith.xori %a, %b : vector<16xi32>
  return %r : vector<16xi32>
}

// Non-512-bit vectors and scalars stay legal on both backends.

// LLVM-LABEL: func @vec_andi_256(
func.func @vec_andi_256(%a: vector<8xi32>, %b: vector<8xi32>) -> vector<8xi32> {
  // LLVM-NOT: aievec.band
  // LLVM: arith.andi %{{.*}}, %{{.*}} : vector<8xi32>
  %r = arith.andi %a, %b : vector<8xi32>
  return %r : vector<8xi32>
}

// LLVM-LABEL: func @scalar_andi(
func.func @scalar_andi(%a: i32, %b: i32) -> i32 {
  // LLVM-NOT: aievec.band
  // LLVM: arith.andi %{{.*}}, %{{.*}} : i32
  %r = arith.andi %a, %b : i32
  return %r : i32
}

// NO-BITWISE-AIEVEC: module
// NO-BITWISE-AIEVEC-NOT: aievec.band
// NO-BITWISE-AIEVEC-NOT: aievec.bor
// NO-BITWISE-AIEVEC-NOT: aievec.bxor
// NO-BITWISE-AIEVEC-NOT: aievec.bneg

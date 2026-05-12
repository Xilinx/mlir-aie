//===- test-int-add-sub-backend.mlir - int add/sub legality per backend -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Integer vector arith.{addi,subi} → aievec.{add_elem,sub_elem} only on the
// CPP (chess) backend; on the LLVMIR (Peano) backend the arith ops must pass
// through unchanged so the standard arith→LLVM lowering can handle them.
//
// Regression for issue #3027: prior to the fix, arith.subi on vectors was
// marked illegal under both backends but had a conversion pattern only under
// CPP, hard-failing aiecc on AIE2P with target-backend=llvmir for vector
// sizes the rewrite would otherwise have matched (e.g. vector<32xi16>).

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=cpp" \
// RUN:   | FileCheck %s --check-prefix=CPP
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2 target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p target-backend=llvmir" \
// RUN:   | FileCheck %s --check-prefix=NO-INT-ADD-SUB-AIEVEC

// CPP-LABEL: func @vec_addi_i16_512(
// LLVM-LABEL: func @vec_addi_i16_512(
func.func @vec_addi_i16_512(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi16> {
  // CPP:  aievec.add_elem %{{.*}}, %{{.*}} : vector<32xi16>
  // LLVM-NOT: aievec.add_elem
  // LLVM:  arith.addi %{{.*}}, %{{.*}} : vector<32xi16>
  %r = arith.addi %a, %b : vector<32xi16>
  return %r : vector<32xi16>
}

// CPP-LABEL: func @vec_subi_i16_512(
// LLVM-LABEL: func @vec_subi_i16_512(
func.func @vec_subi_i16_512(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi16> {
  // CPP:  aievec.sub_elem %{{.*}}, %{{.*}} : vector<32xi16>
  // LLVM-NOT: aievec.sub_elem
  // LLVM:  arith.subi %{{.*}}, %{{.*}} : vector<32xi16>
  %r = arith.subi %a, %b : vector<32xi16>
  return %r : vector<32xi16>
}

// CPP-LABEL: func @vec_subi_i8_512(
// LLVM-LABEL: func @vec_subi_i8_512(
func.func @vec_subi_i8_512(%a: vector<64xi8>, %b: vector<64xi8>) -> vector<64xi8> {
  // CPP:  aievec.sub_elem %{{.*}}, %{{.*}} : vector<64xi8>
  // LLVM-NOT: aievec.sub_elem
  // LLVM:  arith.subi %{{.*}}, %{{.*}} : vector<64xi8>
  %r = arith.subi %a, %b : vector<64xi8>
  return %r : vector<64xi8>
}

// 256-bit i8 vectors stay legal on both backends (no rewrite, no failure).

// CPP-LABEL: func @vec_subi_i8_256(
// LLVM-LABEL: func @vec_subi_i8_256(
func.func @vec_subi_i8_256(%a: vector<32xi8>, %b: vector<32xi8>) -> vector<32xi8> {
  // CPP-NOT: aievec.sub_elem
  // LLVM-NOT: aievec.sub_elem
  // CPP:  arith.subi %{{.*}}, %{{.*}} : vector<32xi8>
  // LLVM: arith.subi %{{.*}}, %{{.*}} : vector<32xi8>
  %r = arith.subi %a, %b : vector<32xi8>
  return %r : vector<32xi8>
}

// Scalars stay legal on both backends.

// CPP-LABEL: func @scalar_subi(
// LLVM-LABEL: func @scalar_subi(
func.func @scalar_subi(%a: i32, %b: i32) -> i32 {
  // CPP-NOT: aievec.sub_elem
  // LLVM-NOT: aievec.sub_elem
  // CPP:  arith.subi %{{.*}}, %{{.*}} : i32
  // LLVM: arith.subi %{{.*}}, %{{.*}} : i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

// NO-INT-ADD-SUB-AIEVEC: module
// NO-INT-ADD-SUB-AIEVEC-NOT: aievec.add_elem
// NO-INT-ADD-SUB-AIEVEC-NOT: aievec.sub_elem

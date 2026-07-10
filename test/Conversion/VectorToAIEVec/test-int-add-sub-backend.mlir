//===- test-int-add-sub-backend.mlir - int add/sub legality per backend -*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" \
// RUN:   | FileCheck %s --check-prefix=NO-INT-ADD-SUB-AIEVEC

// LLVM-LABEL: func @vec_addi_i16_512(
func.func @vec_addi_i16_512(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi16> {
  // LLVM-NOT: aievec.add_elem
  // LLVM:  arith.addi %{{.*}}, %{{.*}} : vector<32xi16>
  %r = arith.addi %a, %b : vector<32xi16>
  return %r : vector<32xi16>
}

// LLVM-LABEL: func @vec_subi_i16_512(
func.func @vec_subi_i16_512(%a: vector<32xi16>, %b: vector<32xi16>) -> vector<32xi16> {
  // LLVM-NOT: aievec.sub_elem
  // LLVM:  arith.subi %{{.*}}, %{{.*}} : vector<32xi16>
  %r = arith.subi %a, %b : vector<32xi16>
  return %r : vector<32xi16>
}

// LLVM-LABEL: func @vec_subi_i8_512(
func.func @vec_subi_i8_512(%a: vector<64xi8>, %b: vector<64xi8>) -> vector<64xi8> {
  // LLVM-NOT: aievec.sub_elem
  // LLVM:  arith.subi %{{.*}}, %{{.*}} : vector<64xi8>
  %r = arith.subi %a, %b : vector<64xi8>
  return %r : vector<64xi8>
}

// 256-bit i8 vectors stay legal on both backends (no rewrite, no failure).

// LLVM-LABEL: func @vec_subi_i8_256(
func.func @vec_subi_i8_256(%a: vector<32xi8>, %b: vector<32xi8>) -> vector<32xi8> {
  // LLVM-NOT: aievec.sub_elem
  // LLVM: arith.subi %{{.*}}, %{{.*}} : vector<32xi8>
  %r = arith.subi %a, %b : vector<32xi8>
  return %r : vector<32xi8>
}

// Scalars stay legal on both backends.

// LLVM-LABEL: func @scalar_subi(
func.func @scalar_subi(%a: i32, %b: i32) -> i32 {
  // LLVM-NOT: aievec.sub_elem
  // LLVM: arith.subi %{{.*}}, %{{.*}} : i32
  %r = arith.subi %a, %b : i32
  return %r : i32
}

// NO-INT-ADD-SUB-AIEVEC: module
// NO-INT-ADD-SUB-AIEVEC-NOT: aievec.add_elem
// NO-INT-ADD-SUB-AIEVEC-NOT: aievec.sub_elem

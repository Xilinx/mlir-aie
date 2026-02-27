//===- test-neg-aie2p.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// Test: aievec.neg → scalar llvm.fneg via extract/insert unrolling

// CHECK-LABEL: @test_neg_v16f32
// CHECK-SAME: %[[ARG:.*]]: vector<16xf32>
func.func @test_neg_v16f32(%arg0 : vector<16xf32>) -> vector<16xf32> {
  // CHECK: llvm.mlir.poison
  // CHECK: llvm.extractelement
  // CHECK: llvm.fneg
  // CHECK: llvm.insertelement
  %0 = aievec.neg %arg0 : vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// CHECK-LABEL: @test_neg_v16bf16
// CHECK-SAME: %[[ARG:.*]]: vector<16xbf16>
func.func @test_neg_v16bf16(%arg0 : vector<16xbf16>) -> vector<16xbf16> {
  // CHECK: llvm.mlir.poison
  // CHECK: llvm.extractelement
  // CHECK: llvm.fneg
  // CHECK: llvm.insertelement
  %0 = aievec.neg %arg0 : vector<16xbf16>
  return %0 : vector<16xbf16>
}

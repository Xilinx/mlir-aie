//===- test-inv-aie2p.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// Test: aievec.inv → xllvm.intr.aie2p.inv for AIE2P

// CHECK-LABEL: @test_inv_f32
// CHECK-SAME: %[[ARG:.*]]: f32
func.func @test_inv_f32(%arg0 : f32) -> f32 {
  // CHECK: %[[INV:.*]] = "xllvm.intr.aie2p.inv"(%[[ARG]]) : (f32) -> f32
  %0 = aievec.inv %arg0 : f32
  // CHECK: return %[[INV]] : f32
  return %0 : f32
}

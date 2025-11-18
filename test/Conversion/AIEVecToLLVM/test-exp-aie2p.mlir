//===- test-exp-aie2p.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

func.func @v16bf16_exp(%arg0 : vector<16xbf16>) -> vector<16xbf16> {
  %0 = aievec.exp %arg0 : vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: @v16bf16_exp
// CHECK-SAME: %[[ARG0:.*]]: vector<16xbf16>
// CHECK: %[[LOG2E:.*]] = llvm.mlir.constant(1.445310e+00 : bf16) : bf16
// CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<1xbf16>
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[INSERT:.*]] = llvm.insertelement %[[LOG2E]], %[[UNDEF]][%[[ZERO]] : i32] : vector<1xbf16>
// CHECK: %[[LOG2EVEC:.*]] = vector.shuffle %[[INSERT]], %[[INSERT]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<1xbf16>, vector<1xbf16>
// CHECK: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
// CHECK: %[[LHSPAD:.*]] = vector.shuffle %[[ARG0]], %[[ARG0]]
// CHECK: %[[RHSPAD:.*]] = vector.shuffle %[[LOG2EVEC]], %[[LOG2EVEC]]
// CHECK: %[[MULRES:.*]] = "xllvm.intr.aie2p.I512.I512.ACC512.bf.mul.conf"(%[[LHSPAD]], %[[RHSPAD]], %[[CONF]]) : (vector<32xbf16>, vector<32xbf16>, i32) -> vector<16xf32>
// CHECK: %[[EXP2:.*]] = "xllvm.intr.aie2p.exp2"(%[[MULRES]]) : (vector<16xf32>) -> vector<16xbf16>
// CHECK: return %[[EXP2]] : vector<16xbf16>

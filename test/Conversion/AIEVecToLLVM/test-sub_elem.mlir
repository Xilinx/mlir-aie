//===- test-sub_elem.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -convert-aievec-to-llvm -split-input-file | FileCheck %s
// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie2p" -split-input-file | FileCheck %s --check-prefix=AIE2P

// CHECK-LABEL: sub_elem_flat_fp32
// CHECK-SAME: %[[LHS:[a-zA-Z0-9]+]]: vector<16xf32>
// CHECK-SAME: %[[RHS:[a-zA-Z0-9]+]]: vector<16xf32>
// AIE2P-LABEL: sub_elem_flat_fp32
// AIE2P-SAME: %[[ARG0:[a-zA-Z0-9]+]]: vector<16xf32>
// AIE2P-SAME: %[[ARG1:[a-zA-Z0-9]+]]: vector<16xf32>
func.func @sub_elem_flat_fp32(%lhs : vector<16xf32>, %rhs : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[CONF:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[LHS_BC:.*]] = llvm.bitcast %[[LHS]] : vector<16xf32> to vector<8xi64>
  // CHECK: %[[RHS_BC:.*]] = llvm.bitcast %[[RHS]] : vector<16xf32> to vector<8xi64>
  // CHECK: %[[SUB:.*]] = "xllvm.intr.aie2.sub.accfloat"(%[[LHS_BC]], %[[RHS_BC]], %[[CONF]]) : (vector<8xi64>, vector<8xi64>, i32) -> vector<8xi64>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[SUB]] : vector<8xi64> to vector<16xf32>
  // AIE2P: %[[BC0:.*]] = llvm.bitcast %[[ARG0]] : vector<16xf32> to vector<8xi64>
  // AIE2P: %[[BC1:.*]] = llvm.bitcast %[[ARG1]] : vector<16xf32> to vector<8xi64>
  // AIE2P: %[[SHUF0:.*]] = vector.shuffle %[[BC0]], %[[BC0]] [0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<8xi64>, vector<8xi64>
  // AIE2P: %[[SHUF1:.*]] = vector.shuffle %[[BC1]], %[[BC1]] [0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] : vector<8xi64>, vector<8xi64>
  // AIE2P: %[[BC2:.*]] = llvm.bitcast %[[SHUF0]] : vector<32xi64> to vector<64xf32>
  // AIE2P: %[[BC3:.*]] = llvm.bitcast %[[SHUF1]] : vector<32xi64> to vector<64xf32>
  // AIE2P: %[[CONF:.*]] = llvm.mlir.constant(60 : i32) : i32
  // AIE2P: %[[SUB:.*]] = "xllvm.intr.aie2p.ACC2048.accfloat.sub.conf"(%[[BC2]], %[[BC3]], %[[CONF]]) : (vector<64xf32>, vector<64xf32>, i32) -> vector<64xf32>
  // AIE2P: %[[BC4:.*]] = llvm.bitcast %[[SUB]] : vector<64xf32> to vector<32xi64>
  // AIE2P: %[[SHUF2:.*]] = vector.shuffle %[[BC4]], %[[BC4]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<32xi64>, vector<32xi64>
  // AIE2P: %[[RES:.*]] = llvm.bitcast %[[SHUF2]] : vector<8xi64> to vector<16xf32>
  %0 = aievec.sub_elem %lhs, %rhs : vector<16xf32>
  return %0 : vector<16xf32>
}

//===- test-ext-elem.mlir ---------------------------------------*- MLIR -*-===//
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

// CHECK-LABEL: ext_elem_i32
// CHECK-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<16xi32>
// CHECK-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
// AIE2P-LABEL: ext_elem_i32
// AIE2P-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<16xi32>
// AIE2P-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
func.func @ext_elem_i32(%vec : vector<16xi32>, %idx : i32) -> i32 {
  // CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[EXT:.*]] = "xllvm.intr.aie2.vextract.elem32.I512"(%[[VEC]], %[[IDX]], %[[SIGN]]) : (vector<16xi32>, i32, i32) -> i32
  // AIE2P: %[[EXT:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : i32] : vector<16xi32>
  %0 = aievec.ext_elem %vec, %idx : vector<16xi32>, i32, i32
  return %0 : i32
}

// -----

// CHECK-LABEL: ext_elem_i16
// CHECK-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<32xi16>
// CHECK-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
// AIE2P-LABEL: ext_elem_i16
// AIE2P-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<32xi16>
// AIE2P-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
func.func @ext_elem_i16(%vec : vector<32xi16>, %idx : i32) -> i16 {
  // CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[EXT:.*]] = "xllvm.intr.aie2.vextract.elem16.I512"(%[[VEC]], %[[IDX]], %[[SIGN]]) : (vector<32xi16>, i32, i32) -> i32
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[EXT]] : i32 to i16
  // AIE2P: %[[EXT:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : i32] : vector<32xi16>
  %0 = aievec.ext_elem %vec, %idx : vector<32xi16>, i32, i16
  return %0 : i16
}

// -----

// CHECK-LABEL: ext_elem_bf16
// CHECK-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<32xbf16>
// CHECK-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
// AIE2P-LABEL: ext_elem_bf16
// AIE2P-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<32xbf16>
// AIE2P-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
func.func @ext_elem_bf16(%vec : vector<32xbf16>, %idx : i32) -> bf16 {
  // CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[BITCAST_VEC:.*]] = llvm.bitcast %[[VEC]] : vector<32xbf16> to vector<32xi16>
  // CHECK: %[[EXT:.*]] = "xllvm.intr.aie2.vextract.elem16.I512"(%[[BITCAST_VEC]], %[[IDX]], %[[SIGN]]) : (vector<32xi16>, i32, i32) -> i32
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[EXT]] : i32 to i16
  // CHECK: %[[BITCAST:.*]] = llvm.bitcast %[[TRUNC]] : i16 to bf16
  // AIE2P: %[[EXT:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : i32] : vector<32xbf16>
  %0 = aievec.ext_elem %vec, %idx : vector<32xbf16>, i32, bf16
  return %0 : bf16
}

// -----

// CHECK-LABEL: ext_elem_i8
// CHECK-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<64xi8>
// CHECK-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
// AIE2P-LABEL: ext_elem_i8
// AIE2P-SAME: %[[VEC:[a-zA-Z0-9]+]]: vector<64xi8>
// AIE2P-SAME: %[[IDX:[a-zA-Z0-9]+]]: i32
func.func @ext_elem_i8(%vec : vector<64xi8>, %idx : i32) -> i8 {
  // CHECK: %[[SIGN:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[EXT:.*]] = "xllvm.intr.aie2.vextract.elem8.I512"(%[[VEC]], %[[IDX]], %[[SIGN]]) : (vector<64xi8>, i32, i32) -> i32
  // CHECK: %[[TRUNC:.*]] = llvm.trunc %[[EXT]] : i32 to i8
  // AIE2P: %[[EXT:.*]] = llvm.extractelement %[[VEC]][%[[IDX]] : i32] : vector<64xi8>
  %0 = aievec.ext_elem %vec, %idx : vector<64xi8>, i32, i8
  return %0 : i8
}

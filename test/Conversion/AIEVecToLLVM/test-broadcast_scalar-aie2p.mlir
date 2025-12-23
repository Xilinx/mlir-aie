//===- test-broadcast_scalar-aie2p.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

func.func @i8_broadcast_scalar_aie2p(%arg0 : i8) -> vector<64xi8> {
  %0 = aievec.broadcast_scalar %arg0 : i8, vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @i8_broadcast_scalar_aie2p
// CHECK-SAME: %[[ARG0:.*]]: i8
// CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<64xi8>
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[INSERT:.*]] = llvm.insertelement %[[ARG0]], %[[POISON]][%[[IDX0]] : i64] : vector<64xi8>
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.shuffle %[[INSERT]], %[[INSERT]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<64xi8>, vector<64xi8>
// CHECK-NEXT: return %[[BROADCAST]] : vector<64xi8>

// -----

func.func @i16_broadcast_scalar_aie2p(%arg0 : i16) -> vector<32xi16> {
  %0 = aievec.broadcast_scalar %arg0 : i16, vector<32xi16>
  return %0 : vector<32xi16>
}

// CHECK-LABEL: @i16_broadcast_scalar_aie2p
// CHECK-SAME: %[[ARG0:.*]]: i16
// CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<32xi16>
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[INSERT:.*]] = llvm.insertelement %[[ARG0]], %[[POISON]][%[[IDX0]] : i64] : vector<32xi16>
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.shuffle %[[INSERT]], %[[INSERT]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<32xi16>, vector<32xi16>
// CHECK-NEXT: return %[[BROADCAST]] : vector<32xi16>

// -----

func.func @i32_broadcast_scalar_aie2p(%arg0 : i32) -> vector<16xi32> {
  %0 = aievec.broadcast_scalar %arg0 : i32, vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @i32_broadcast_scalar_aie2p
// CHECK-SAME: %[[ARG0:.*]]: i32
// CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<16xi32>
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[INSERT:.*]] = llvm.insertelement %[[ARG0]], %[[POISON]][%[[IDX0]] : i64] : vector<16xi32>
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.shuffle %[[INSERT]], %[[INSERT]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xi32>, vector<16xi32>
// CHECK-NEXT: return %[[BROADCAST]] : vector<16xi32>

// -----

func.func @bf16_broadcast_scalar_aie2p(%arg0 : bf16) -> vector<32xbf16> {
  %0 = aievec.broadcast_scalar %arg0 : bf16, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// CHECK-LABEL: @bf16_broadcast_scalar_aie2p
// CHECK-SAME: %[[ARG0:.*]]: bf16
// CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<32xbf16>
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[INSERT:.*]] = llvm.insertelement %[[ARG0]], %[[POISON]][%[[IDX0]] : i64] : vector<32xbf16>
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.shuffle %[[INSERT]], %[[INSERT]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: return %[[BROADCAST]] : vector<32xbf16>

// -----

func.func @f32_broadcast_scalar_aie2p(%arg0 : f32) -> vector<16xf32> {
  %0 = aievec.broadcast_scalar %arg0 : f32, vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @f32_broadcast_scalar_aie2p
// CHECK-SAME: %[[ARG0:.*]]: f32
// CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<16xf32>
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[INSERT:.*]] = llvm.insertelement %[[ARG0]], %[[POISON]][%[[IDX0]] : i64] : vector<16xf32>
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.shuffle %[[INSERT]], %[[INSERT]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: return %[[BROADCAST]] : vector<16xf32>

// -----

func.func @bf16_broadcast_scalar_256bit_aie2p(%arg0 : bf16) -> vector<16xbf16> {
  %0 = aievec.broadcast_scalar %arg0 : bf16, vector<16xbf16>
  return %0 : vector<16xbf16>
}

// CHECK-LABEL: @bf16_broadcast_scalar_256bit_aie2p
// CHECK-SAME: %[[ARG0:.*]]: bf16
// CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<16xbf16>
// CHECK-NEXT: %[[IDX0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: %[[INSERT:.*]] = llvm.insertelement %[[ARG0]], %[[POISON]][%[[IDX0]] : i64] : vector<16xbf16>
// CHECK-NEXT: %[[BROADCAST:.*]] = vector.shuffle %[[INSERT]], %[[INSERT]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<16xbf16>, vector<16xbf16>
// CHECK-NEXT: return %[[BROADCAST]] : vector<16xbf16>

//===- one-lane-insert-to-bitcast.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --aievec-one-lane-insert-to-bitcast | FileCheck %s

// CHECK-LABEL: llvm.func @lane0_f32
// CHECK-SAME: %[[V:.*]]: vector<16xf32>
// CHECK: %[[E:.*]] = llvm.extractelement %[[V]]
// CHECK: %[[R:.*]] = llvm.bitcast %[[E]] : f32 to vector<1xf32>
// CHECK-NOT: llvm.insertelement
// CHECK: llvm.return %[[R]] : vector<1xf32>
llvm.func @lane0_f32(%v: vector<16xf32>) -> vector<1xf32> {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %e = llvm.extractelement %v[%c0 : i64] : vector<16xf32>
  %p = llvm.mlir.poison : vector<1xf32>
  %c0_i32 = llvm.mlir.constant(0 : i32) : i32
  %r = llvm.insertelement %e, %p[%c0_i32 : i32] : vector<1xf32>
  llvm.return %r : vector<1xf32>
}

// -----

// The vector inserted into does not matter: the insert replaces its only lane.
// CHECK-LABEL: llvm.func @bf16_into_value
// CHECK-SAME: %[[S:.*]]: bf16, %{{.*}}: vector<1xbf16>
// CHECK: %[[R:.*]] = llvm.bitcast %[[S]] : bf16 to vector<1xbf16>
// CHECK: llvm.return %[[R]] : vector<1xbf16>
llvm.func @bf16_into_value(%s: bf16, %v: vector<1xbf16>) -> vector<1xbf16> {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %r = llvm.insertelement %s, %v[%c0 : i64] : vector<1xbf16>
  llvm.return %r : vector<1xbf16>
}

// -----

// CHECK-LABEL: llvm.func @two_lanes
// CHECK: llvm.insertelement
// CHECK-NOT: llvm.bitcast
llvm.func @two_lanes(%s: f32, %v: vector<2xf32>) -> vector<2xf32> {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %r = llvm.insertelement %s, %v[%c0 : i64] : vector<2xf32>
  llvm.return %r : vector<2xf32>
}

// -----

// A position that is not a constant 0 is left alone.
// CHECK-LABEL: llvm.func @dynamic_position
// CHECK: llvm.insertelement
// CHECK-NOT: llvm.bitcast
llvm.func @dynamic_position(%s: f32, %v: vector<1xf32>, %i: i32) -> vector<1xf32> {
  %r = llvm.insertelement %s, %v[%i : i32] : vector<1xf32>
  llvm.return %r : vector<1xf32>
}

// -----

// CHECK-LABEL: llvm.func @scalable
// CHECK: llvm.insertelement
// CHECK-NOT: llvm.bitcast
llvm.func @scalable(%s: f32, %v: vector<[1]xf32>) -> vector<[1]xf32> {
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %r = llvm.insertelement %s, %v[%c0 : i64] : vector<[1]xf32>
  llvm.return %r : vector<[1]xf32>
}

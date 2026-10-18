//===- test-one-lane-vector.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s
// RUN: aie-opt %s -split-input-file --convert-aievec-to-llvm="aie-target=aie2" | FileCheck %s

// CHECK-LABEL: func @broadcast_f32
// CHECK-SAME: %[[S:.*]]: f32
// CHECK: %[[R:.*]] = llvm.bitcast %[[S]] : f32 to vector<1xf32>
// CHECK: return %[[R]] : vector<1xf32>
func.func @broadcast_f32(%s: f32) -> vector<1xf32> {
  %0 = vector.broadcast %s : f32 to vector<1xf32>
  return %0 : vector<1xf32>
}

// -----

// CHECK-LABEL: func @broadcast_bf16
// CHECK-SAME: %[[S:.*]]: bf16
// CHECK: %[[R:.*]] = llvm.bitcast %[[S]] : bf16 to vector<1xbf16>
// CHECK: return %[[R]] : vector<1xbf16>
func.func @broadcast_bf16(%s: bf16) -> vector<1xbf16> {
  %0 = vector.broadcast %s : bf16 to vector<1xbf16>
  return %0 : vector<1xbf16>
}

// -----

// CHECK-LABEL: func @insert_f32
// CHECK-SAME: %[[S:.*]]: f32, %{{.*}}: vector<1xf32>
// CHECK: %[[R:.*]] = llvm.bitcast %[[S]] : f32 to vector<1xf32>
// CHECK: return %[[R]] : vector<1xf32>
func.func @insert_f32(%s: f32, %v: vector<1xf32>) -> vector<1xf32> {
  %0 = vector.insert %s, %v [0] : f32 into vector<1xf32>
  return %0 : vector<1xf32>
}

// -----

// Wider results, vector sources, dynamic positions and scalable vectors are
// left to the vector lowering.
// CHECK-LABEL: func @unchanged
// CHECK-NOT: llvm.bitcast
// CHECK: vector.broadcast %{{.*}} : f32 to vector<2xf32>
// CHECK: vector.broadcast %{{.*}} : vector<1xf32> to vector<1xf32>
// CHECK: vector.insert %{{.*}}, %{{.*}} [%{{.*}}] : f32 into vector<1xf32>
// CHECK: vector.broadcast %{{.*}} : f32 to vector<[1]xf32>
func.func @unchanged(%s: f32, %v: vector<1xf32>, %i: index)
    -> (vector<2xf32>, vector<1xf32>, vector<1xf32>, vector<[1]xf32>) {
  %0 = vector.broadcast %s : f32 to vector<2xf32>
  %1 = vector.broadcast %v : vector<1xf32> to vector<1xf32>
  %2 = vector.insert %s, %v [%i] : f32 into vector<1xf32>
  %3 = vector.broadcast %s : f32 to vector<[1]xf32>
  return %0, %1, %2, %3 : vector<2xf32>, vector<1xf32>, vector<1xf32>,
      vector<[1]xf32>
}

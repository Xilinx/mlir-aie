//===- test-iter-args-flatten-aie2p.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2p" | FileCheck %s

// A loop that carries an n-D vector carries it flat, and the flattened
// elementwise op in its body works on the loop-carried value with no shape
// cast in between.
// CHECK-LABEL: func @iter
// CHECK-SAME: %{{.*}}: memref<4x2x16xbf16>, %[[INIT:[a-zA-Z0-9]+]]: vector<2x16xbf16>
// CHECK: %[[FLAT:.*]] = vector.shape_cast %[[INIT]] : vector<2x16xbf16> to vector<32xbf16>
// CHECK: %[[R:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[FLAT]]) -> (vector<32xbf16>) {
// CHECK-NOT: vector.shape_cast
// CHECK: aievec.ext %[[ACC]] {index = 0 : i8} : vector<32xbf16>, vector<16xbf16>
// CHECK: aievec.ext %[[ACC]] {index = 1 : i8} : vector<32xbf16>, vector<16xbf16>
// CHECK-NOT: vector.shape_cast
// CHECK: %[[PROD:.*]] = aievec.concat %{{.*}}, %{{.*}} : vector<16xbf16>, vector<32xbf16>
// CHECK-NEXT: scf.yield %[[PROD]] : vector<32xbf16>
// CHECK: vector.shape_cast %[[R]] : vector<32xbf16> to vector<2x16xbf16>
func.func @iter(%m: memref<4x2x16xbf16>, %init: vector<2x16xbf16>) -> vector<2x16xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f0 = arith.constant 0.0 : bf16
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (vector<2x16xbf16>) {
    %v = vector.transfer_read %m[%i, %c0, %c0], %f0 {in_bounds = [true, true]} : memref<4x2x16xbf16>, vector<2x16xbf16>
    %s = arith.mulf %acc, %v : vector<2x16xbf16>
    scf.yield %s : vector<2x16xbf16>
  }
  return %r : vector<2x16xbf16>
}

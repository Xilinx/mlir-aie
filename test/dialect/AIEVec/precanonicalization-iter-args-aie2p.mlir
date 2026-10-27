//===- precanonicalization-iter-args-aie2p.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -canonicalize-vector-for-aievec="aie-target=aie2p" -split-input-file | FileCheck %s
// RUN: aie-opt %s -canonicalize-vector-for-aievec="aie-target=aie2" -split-input-file | FileCheck %s --check-prefix=AIE2

// On AIE2P, an scf.for carries each n-D vector iter_arg as a rank-1 vector,
// with shape casts at the loop boundary. AIE2 keeps the loops as they are.

// CHECK-LABEL: func.func @rank2
// CHECK-SAME: %[[INIT:.*]]: vector<4x16xf32>, %[[X:.*]]: vector<4x16xf32>
// CHECK: %[[FLAT:.*]] = vector.shape_cast %[[INIT]] : vector<4x16xf32> to vector<64xf32>
// CHECK: %[[LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[FLAT]]) -> (vector<64xf32>) {
// CHECK:   %[[ACC2D:.*]] = vector.shape_cast %[[ACC]] : vector<64xf32> to vector<4x16xf32>
// CHECK:   %[[SUM:.*]] = arith.addf %[[ACC2D]], %[[X]] : vector<4x16xf32>
// CHECK:   %[[SUMFLAT:.*]] = vector.shape_cast %[[SUM]] : vector<4x16xf32> to vector<64xf32>
// CHECK:   scf.yield %[[SUMFLAT]] : vector<64xf32>
// CHECK: }
// CHECK: %[[RES:.*]] = vector.shape_cast %[[LOOP]] : vector<64xf32> to vector<4x16xf32>
// CHECK: return %[[RES]] : vector<4x16xf32>
// AIE2-LABEL: func.func @rank2
// AIE2-NOT: vector.shape_cast
// AIE2: scf.for {{.*}} -> (vector<4x16xf32>)
// AIE2-NOT: vector.shape_cast
func.func @rank2(%init: vector<4x16xf32>, %x: vector<4x16xf32>) -> vector<4x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (vector<4x16xf32>) {
    %s = arith.addf %acc, %x : vector<4x16xf32>
    scf.yield %s : vector<4x16xf32>
  }
  return %r : vector<4x16xf32>
}

// -----

// Rank 3 and a leading unit dimension are flattened as well.
// CHECK-LABEL: func.func @rank3_and_unit_dim
// CHECK: vector.shape_cast %{{.*}} : vector<2x2x8xbf16> to vector<32xbf16>
// CHECK: vector.shape_cast %{{.*}} : vector<1x16xi32> to vector<16xi32>
// CHECK: scf.for {{.*}} -> (vector<32xbf16>, vector<16xi32>) {
// CHECK:   scf.yield %{{.*}}, %{{.*}} : vector<32xbf16>, vector<16xi32>
// CHECK: }
// AIE2-LABEL: func.func @rank3_and_unit_dim
// AIE2: scf.for {{.*}} -> (vector<2x2x8xbf16>, vector<1x16xi32>)
func.func @rank3_and_unit_dim(%a: vector<2x2x8xbf16>, %b: vector<1x16xi32>)
    -> (vector<2x2x8xbf16>, vector<1x16xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r:2 = scf.for %i = %c0 to %c8 step %c1 iter_args(%x = %a, %y = %b)
      -> (vector<2x2x8xbf16>, vector<1x16xi32>) {
    %s = arith.addf %x, %x : vector<2x2x8xbf16>
    %t = arith.addi %y, %y : vector<1x16xi32>
    scf.yield %s, %t : vector<2x2x8xbf16>, vector<1x16xi32>
  }
  return %r#0, %r#1 : vector<2x2x8xbf16>, vector<1x16xi32>
}

// -----

// Only the n-D vector iter_args change; scalars, rank-1 vectors and scalable
// vectors keep their types and positions.
// CHECK-LABEL: func.func @mixed
// CHECK-SAME: %[[S:[a-zA-Z0-9]+]]: f32, %[[V:[a-zA-Z0-9]+]]: vector<16xf32>, %[[M:[a-zA-Z0-9]+]]: vector<2x16xf32>, %[[SC:[a-zA-Z0-9]+]]: vector<2x[8]xf32>
// CHECK: %[[MFLAT:.*]] = vector.shape_cast %[[M]] : vector<2x16xf32> to vector<32xf32>
// CHECK: %[[LOOP:.*]]:5 = scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[A0:.*]] = %[[S]], %[[A1:.*]] = %[[V]], %[[A2:.*]] = %[[MFLAT]], %[[A3:.*]] = %[[SC]], %[[A4:.*]] = %{{.*}}) -> (f32, vector<16xf32>, vector<32xf32>, vector<2x[8]xf32>, index) {
// CHECK:   %[[A2ND:.*]] = vector.shape_cast %[[A2]] : vector<32xf32> to vector<2x16xf32>
// CHECK:   %[[N2:.*]] = arith.addf %[[A2ND]], %[[A2ND]] : vector<2x16xf32>
// CHECK:   %[[N2FLAT:.*]] = vector.shape_cast %[[N2]] : vector<2x16xf32> to vector<32xf32>
// CHECK:   scf.yield %{{.*}}, %{{.*}}, %[[N2FLAT]], %{{.*}}, %[[IV]] : f32, vector<16xf32>, vector<32xf32>, vector<2x[8]xf32>, index
// CHECK: }
// CHECK: %[[R2:.*]] = vector.shape_cast %[[LOOP]]#2 : vector<32xf32> to vector<2x16xf32>
// CHECK: return %[[LOOP]]#0, %[[LOOP]]#1, %[[R2]], %[[LOOP]]#3, %[[LOOP]]#4
func.func @mixed(%s: f32, %v: vector<16xf32>, %m: vector<2x16xf32>,
                 %sc: vector<2x[8]xf32>)
    -> (f32, vector<16xf32>, vector<2x16xf32>, vector<2x[8]xf32>, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r:5 = scf.for %i = %c0 to %c8 step %c1
      iter_args(%a0 = %s, %a1 = %v, %a2 = %m, %a3 = %sc, %a4 = %c0)
      -> (f32, vector<16xf32>, vector<2x16xf32>, vector<2x[8]xf32>, index) {
    %n0 = arith.addf %a0, %a0 : f32
    %n1 = arith.addf %a1, %a1 : vector<16xf32>
    %n2 = arith.addf %a2, %a2 : vector<2x16xf32>
    %n3 = arith.addf %a3, %a3 : vector<2x[8]xf32>
    scf.yield %n0, %n1, %n2, %n3, %i
        : f32, vector<16xf32>, vector<2x16xf32>, vector<2x[8]xf32>, index
  }
  return %r#0, %r#1, %r#2, %r#3, %r#4
      : f32, vector<16xf32>, vector<2x16xf32>, vector<2x[8]xf32>, index
}

// -----

// In a loop nest, the inner loop takes the outer loop's flat iter_arg
// directly: the shape casts between the two loops fold away.
// CHECK-LABEL: func.func @nested
// CHECK-SAME: %[[INIT:.*]]: vector<4x8xf32>
// CHECK: %[[FLAT:.*]] = vector.shape_cast %[[INIT]] : vector<4x8xf32> to vector<32xf32>
// CHECK: %[[OUTER:.*]] = scf.for {{.*}} iter_args(%[[OACC:.*]] = %[[FLAT]]) -> (vector<32xf32>) {
// CHECK-NOT: vector.shape_cast
// CHECK:   %[[INNER:.*]] = scf.for {{.*}} iter_args(%[[IACC:.*]] = %[[OACC]]) -> (vector<32xf32>) {
// CHECK:     %[[IACC2D:.*]] = vector.shape_cast %[[IACC]] : vector<32xf32> to vector<4x8xf32>
// CHECK:     %[[SUM:.*]] = arith.addf %[[IACC2D]], %[[IACC2D]] : vector<4x8xf32>
// CHECK:     %[[SUMFLAT:.*]] = vector.shape_cast %[[SUM]] : vector<4x8xf32> to vector<32xf32>
// CHECK:     scf.yield %[[SUMFLAT]] : vector<32xf32>
// CHECK:   }
// CHECK-NOT: vector.shape_cast
// CHECK:   scf.yield %[[INNER]] : vector<32xf32>
// CHECK: }
// CHECK: %[[RES:.*]] = vector.shape_cast %[[OUTER]] : vector<32xf32> to vector<4x8xf32>
// CHECK: return %[[RES]]
// AIE2-LABEL: func.func @nested
// AIE2-NOT: vector.shape_cast
func.func @nested(%init: vector<4x8xf32>) -> vector<4x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%oacc = %init) -> (vector<4x8xf32>) {
    %in = scf.for %j = %c0 to %c4 step %c1 iter_args(%iacc = %oacc) -> (vector<4x8xf32>) {
      %s = arith.addf %iacc, %iacc : vector<4x8xf32>
      scf.yield %s : vector<4x8xf32>
    }
    scf.yield %in : vector<4x8xf32>
  }
  return %r : vector<4x8xf32>
}

// -----

// Loops without an n-D vector iter_arg are left as they are.
// CHECK-LABEL: func.func @no_nd_iter_args
// CHECK-NOT: vector.shape_cast
// CHECK: scf.for {{.*}} -> (vector<16xf32>, f32) {
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK-NOT: vector.shape_cast
// CHECK: return
func.func @no_nd_iter_args(%v: vector<16xf32>, %s: f32, %x: vector<2x8xf32>,
                           %m: memref<8x2x8xf32>)
    -> (vector<16xf32>, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r:2 = scf.for %i = %c0 to %c8 step %c1 iter_args(%a = %v, %b = %s)
      -> (vector<16xf32>, f32) {
    %n = arith.addf %a, %a : vector<16xf32>
    scf.yield %n, %b : vector<16xf32>, f32
  }
  scf.for %i = %c0 to %c8 step %c1 {
    memref.store %s, %m[%i, %c0, %c0] : memref<8x2x8xf32>
  }
  return %r#0, %r#1 : vector<16xf32>, f32
}

// -----

// The loop's attributes and unsigned comparison are kept.
// CHECK-LABEL: func.func @attrs
// CHECK: scf.for unsigned {{.*}} -> (vector<32xf32>) {
// CHECK: } {some_attr}
func.func @attrs(%init: vector<2x16xf32>, %lb: index, %ub: index) -> vector<2x16xf32> {
  %c1 = arith.constant 1 : index
  %r = scf.for unsigned %i = %lb to %ub step %c1 iter_args(%acc = %init) -> (vector<2x16xf32>) {
    %s = arith.addf %acc, %acc : vector<2x16xf32>
    scf.yield %s : vector<2x16xf32>
  } {some_attr}
  return %r : vector<2x16xf32>
}

// -----

// An iter_arg the loop yields unchanged needs no shape cast inside the loop.
// CHECK-LABEL: func.func @passthrough
// CHECK: %[[FLAT:.*]] = vector.shape_cast %{{.*}} : vector<2x16xf32> to vector<32xf32>
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[FLAT]]) -> (vector<32xf32>) {
// CHECK-NEXT: scf.yield %[[ACC]] : vector<32xf32>
func.func @passthrough(%init: vector<2x16xf32>) -> vector<2x16xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %init) -> (vector<2x16xf32>) {
    scf.yield %acc : vector<2x16xf32>
  }
  return %r : vector<2x16xf32>
}

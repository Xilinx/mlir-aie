//===- test-native-lanes-cmp-sel-aie2p.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file --convert-vector-to-aievec="aie-target=aie2p" --convert-aievec-to-llvm="aie-target=aie2p" | FileCheck %s

// Compares and selects split into parts in VectorToAIEVec, wide vectors split
// into more than nine parts, and f32 adds split into 16-lane parts lower to
// LLVM with no aievec op left.

// CHECK-LABEL: func.func @cmpf_select_v64bf16
// CHECK-NOT: aievec
// CHECK-COUNT-2: xllvm.intr.aie2p.vsel16
// CHECK-NOT: aievec
// CHECK: return
func.func @cmpf_select_v64bf16(%a: vector<64xbf16>, %b: vector<64xbf16>,
                               %x: vector<64xbf16>, %y: vector<64xbf16>) -> vector<64xbf16> {
  %c = arith.cmpf ogt, %a, %b : vector<64xbf16>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// CHECK-LABEL: func.func @cmpf_v32bf16_select_v32i32
// CHECK-NOT: aievec
// CHECK-COUNT-2: xllvm.intr.aie2p.vsel32
// CHECK-NOT: aievec
// CHECK: return
func.func @cmpf_v32bf16_select_v32i32(%a: vector<32xbf16>, %b: vector<32xbf16>,
                                      %x: vector<32xi32>, %y: vector<32xi32>) -> vector<32xi32> {
  %c = arith.cmpf olt, %a, %b : vector<32xbf16>
  %0 = arith.select %c, %x, %y : vector<32xi1>, vector<32xi32>
  return %0 : vector<32xi32>
}

// -----

// The 16-lane i16 parts of the compare are padded to 32 lanes; the low 16
// bits of its mask select the 16 lanes of i32.
// CHECK-LABEL: func.func @cmpi_v64i16_select_v64i32
// CHECK-SAME: (%[[A:.*]]: vector<64xi16>, %[[B:.*]]: vector<64xi16>, %[[X:.*]]: vector<64xi32>, %[[Y:.*]]: vector<64xi32>)
// CHECK: %[[A0:.*]] = vector.shuffle %[[A]], %[[A]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xi16>, vector<64xi16>
// CHECK: %[[B0:.*]] = vector.shuffle %[[B]], %[[B]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xi16>, vector<64xi16>
// CHECK: %[[PA:.*]] = vector.shuffle %[[A0]], %{{.*}} [0, 1, {{.*}}, 31] : vector<16xi16>, vector<16xi16>
// CHECK: %[[PB:.*]] = vector.shuffle %[[B0]], %{{.*}} [0, 1, {{.*}}, 31] : vector<16xi16>, vector<16xi16>
// CHECK: %[[LT:.*]] = "xllvm.intr.aie2p.vmax.lt16"(%[[PB]], %[[PA]], %{{.*}}) : (vector<32xi16>, vector<32xi16>, i32) -> !llvm.struct<(vector<32xi16>, i32)>
// CHECK: %[[M:.*]] = llvm.extractvalue %[[LT]][1]
// CHECK: %[[X0:.*]] = vector.shuffle %[[X]], %[[X]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xi32>, vector<64xi32>
// CHECK: %[[Y0:.*]] = vector.shuffle %[[Y]], %[[Y]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xi32>, vector<64xi32>
// CHECK: "xllvm.intr.aie2p.vsel32"(%[[Y0]], %[[X0]], %[[M]]) : (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK-COUNT-3: xllvm.intr.aie2p.vsel32
// CHECK-NOT: aievec
// CHECK: return
func.func @cmpi_v64i16_select_v64i32(%a: vector<64xi16>, %b: vector<64xi16>,
                                     %x: vector<64xi32>, %y: vector<64xi32>) -> vector<64xi32> {
  %c = arith.cmpi sgt, %a, %b : vector<64xi16>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// CHECK-LABEL: func.func @cmpi_v64i32_select_v64bf16
// CHECK-NOT: aievec
// CHECK-COUNT-4: xllvm.intr.aie2p.vsel16
// CHECK-NOT: aievec
// CHECK: return
func.func @cmpi_v64i32_select_v64bf16(%a: vector<64xi32>, %b: vector<64xi32>,
                                      %x: vector<64xbf16>, %y: vector<64xbf16>) -> vector<64xbf16> {
  %c = arith.cmpi slt, %a, %b : vector<64xi32>
  %0 = arith.select %c, %x, %y : vector<64xi1>, vector<64xbf16>
  return %0 : vector<64xbf16>
}

// -----

// Part 15 of 16 is lanes 480 to 511.
// CHECK-LABEL: func.func @maximumf_v512bf16
// CHECK-NOT: aievec
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [480, 481, {{.*}}, 511] : vector<512xbf16>, vector<512xbf16>
// CHECK-NOT: aievec
// CHECK: return
func.func @maximumf_v512bf16(%a: vector<512xbf16>, %b: vector<512xbf16>) -> vector<512xbf16> {
  %0 = arith.maximumf %a, %b : vector<512xbf16>
  return %0 : vector<512xbf16>
}

// -----

// 48 lanes of f32 add and sub are three 16-lane parts, each padded to the
// 64-lane accumulator intrinsic.
// CHECK-LABEL: func.func @addf_subf_v48f32
// CHECK-NOT: aievec
// CHECK-COUNT-3: xllvm.intr.aie2p.ACC2048.accfloat.add.conf
// CHECK-COUNT-3: xllvm.intr.aie2p.ACC2048.accfloat.sub.conf
// CHECK-NOT: aievec
// CHECK-NOT: arith.addf
// CHECK-NOT: arith.subf
// CHECK: return
func.func @addf_subf_v48f32(%a: vector<48xf32>, %b: vector<48xf32>)
    -> (vector<48xf32>, vector<48xf32>) {
  %0 = arith.addf %a, %b : vector<48xf32>
  %1 = arith.subf %a, %b : vector<48xf32>
  return %0, %1 : vector<48xf32>, vector<48xf32>
}

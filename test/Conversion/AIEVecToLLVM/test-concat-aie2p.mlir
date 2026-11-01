//===- test-concat-aie2p.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -convert-aievec-to-llvm="aie-target=aie2p" -split-input-file | FileCheck %s

// AIE2p joins the sources of aievec.concat two at a time with vector.shuffle.
// When a round has an odd number, the last one is widened with poison to the
// width of the others first, and only its own lanes are taken.

// CHECK-LABEL: @concat_3xv32bf16
// CHECK-SAME: %[[A:.*]]: vector<32xbf16>, %[[B:.*]]: vector<32xbf16>, %[[C:.*]]: vector<32xbf16>
// CHECK: %[[AB:.*]] = vector.shuffle %[[A]], %[[B]] [0, 1, {{([0-9]+, )+}}63] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[CP:.*]] = vector.shuffle %[[C]], %[[C]] [0, 1, {{([0-9]+, )+}}31, -1, {{(-1, )+}}-1] : vector<32xbf16>, vector<32xbf16>
// CHECK: %[[R:.*]] = vector.shuffle %[[AB]], %[[CP]] [0, 1, {{([0-9]+, )+}}95] : vector<64xbf16>, vector<64xbf16>
// CHECK: return %[[R]] : vector<96xbf16>
func.func @concat_3xv32bf16(%a: vector<32xbf16>, %b: vector<32xbf16>,
                            %c: vector<32xbf16>) -> vector<96xbf16> {
  %0 = aievec.concat %a, %b, %c : vector<32xbf16>, vector<96xbf16>
  return %0 : vector<96xbf16>
}

// -----

// CHECK-LABEL: @concat_4xv16i32
// CHECK-SAME: %[[A:.*]]: vector<16xi32>, %[[B:.*]]: vector<16xi32>, %[[C:.*]]: vector<16xi32>, %[[D:.*]]: vector<16xi32>
// CHECK: %[[AB:.*]] = vector.shuffle %[[A]], %[[B]] [0, 1, {{([0-9]+, )+}}31] : vector<16xi32>, vector<16xi32>
// CHECK: %[[CD:.*]] = vector.shuffle %[[C]], %[[D]] [0, 1, {{([0-9]+, )+}}31] : vector<16xi32>, vector<16xi32>
// CHECK: %[[R:.*]] = vector.shuffle %[[AB]], %[[CD]] [0, 1, {{([0-9]+, )+}}63] : vector<32xi32>, vector<32xi32>
// CHECK: return %[[R]] : vector<64xi32>
func.func @concat_4xv16i32(%a: vector<16xi32>, %b: vector<16xi32>,
                           %c: vector<16xi32>, %d: vector<16xi32>) -> vector<64xi32> {
  %0 = aievec.concat %a, %b, %c, %d : vector<16xi32>, vector<64xi32>
  return %0 : vector<64xi32>
}

// -----

// Eight sources take three rounds.
// CHECK-LABEL: @concat_8xv8f32
// CHECK-COUNT-4: vector.shuffle %{{.*}}, %{{.*}} [0, 1, {{([0-9]+, )+}}15] : vector<8xf32>, vector<8xf32>
// CHECK-COUNT-2: vector.shuffle %{{.*}}, %{{.*}} [0, 1, {{([0-9]+, )+}}31] : vector<16xf32>, vector<16xf32>
// CHECK: %[[R:.*]] = vector.shuffle %{{.*}}, %{{.*}} [0, 1, {{([0-9]+, )+}}63] : vector<32xf32>, vector<32xf32>
// CHECK: return %[[R]] : vector<64xf32>
func.func @concat_8xv8f32(%a: vector<8xf32>) -> vector<64xf32> {
  %0 = aievec.concat %a, %a, %a, %a, %a, %a, %a, %a : vector<8xf32>, vector<64xf32>
  return %0 : vector<64xf32>
}

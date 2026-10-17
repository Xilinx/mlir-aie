//===- test-replicate-shuffle-aie2p.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec='aie-target=aie2p' -cse | FileCheck %s

// A 128-bit vector repeated to 1024 bits: doubled, padded to 512 bits,
// repeated by a t256_2x2_lo shuffle, then concatenated.
// CHECK-LABEL: @repeat_v8bf16_to_v64(
// CHECK-SAME:  %[[A:.*]]: vector<8xbf16>
func.func @repeat_v8bf16_to_v64(%a: vector<8xbf16>) -> vector<64xbf16> {
  // CHECK: %[[D:.*]] = vector.shuffle %[[A]], %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
  // CHECK: %[[P:.*]] = vector.shuffle %[[D]], %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1,
  // CHECK: %[[W:.*]] = vector.bitcast %[[P]] : vector<32xbf16> to vector<2xi256>
  // CHECK: %[[X:.*]] = aievec.shuffle %[[W]], %[[W]] [t256_2x2_lo] : vector<2xi256>
  // CHECK: %[[S:.*]] = vector.bitcast %[[X]] : vector<2xi256> to vector<32xbf16>
  // CHECK: %[[R:.*]] = aievec.concat %[[S]], %[[S]] : vector<32xbf16>, vector<64xbf16>
  // CHECK: return %[[R]]
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
  return %r : vector<64xbf16>
}

// To 512 bits: the shuffle alone.
// CHECK-LABEL: @repeat_v8bf16_to_v32
// CHECK: aievec.shuffle %{{.*}}, %{{.*}} [t256_2x2_lo] : vector<2xi256>
// CHECK: %[[S:.*]] = vector.bitcast
// CHECK-NOT: aievec.concat
// CHECK: return %[[S]]
func.func @repeat_v8bf16_to_v32(%a: vector<8xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
  return %r : vector<32xbf16>
}

// Not a repetition: left as it is.
// CHECK-LABEL: @not_repeat
// CHECK: vector.shuffle %{{.*}}, %{{.*}} [1, 0, 2, 3, 4, 5, 6, 7, 0, 1,
// CHECK-NOT: aievec
func.func @not_repeat(%a: vector<8xbf16>) -> vector<32xbf16> {
  %r = vector.shuffle %a, %a [1, 0, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7] : vector<8xbf16>, vector<8xbf16>
  return %r : vector<32xbf16>
}

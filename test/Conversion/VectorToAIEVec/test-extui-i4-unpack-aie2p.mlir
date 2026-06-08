//===- test-extui-i4-unpack-aie2p.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-vector-to-aievec='aie-target=aie2p target-backend=llvmir' -cse | FileCheck %s

// Canonical standard-MLIR for AWQ int4 unpack on AIE2P:
//   1. byte load (vector<N/2 x i8>)
//   2. vector.bitcast to vector<N x i4> (logical nibble view)
//   3. arith.extui to vector<N x i8>
// Lowers to aievec.unpack on the byte-packed source. The bitcast and extui
// are folded away; vector<N x i4> never reaches Peano (which cannot
// legalize llvm.bitcast on i4 vector types).

// CHECK-LABEL: @extui_i4_to_i8_v64(
// CHECK-SAME:  %[[ARG:.*]]: vector<32xi8>
func.func @extui_i4_to_i8_v64(%pk_i8: vector<32xi8>) -> vector<64xi8> {
  // CHECK:  %[[OUT:.*]] = aievec.unpack %[[ARG]] : vector<32xi8>, vector<64xi8>
  %pk_i4 = vector.bitcast %pk_i8 : vector<32xi8> to vector<64xi4>
  %w_i8 = arith.extui %pk_i4 : vector<64xi4> to vector<64xi8>
  // CHECK: return %[[OUT]] : vector<64xi8>
  return %w_i8 : vector<64xi8>
}

// CHECK-LABEL: @extui_i4_to_i8_v128(
// CHECK-SAME:  %[[ARG:.*]]: vector<64xi8>
func.func @extui_i4_to_i8_v128(%pk_i8: vector<64xi8>) -> vector<128xi8> {
  // CHECK:  %[[OUT:.*]] = aievec.unpack %[[ARG]] : vector<64xi8>, vector<128xi8>
  %pk_i4 = vector.bitcast %pk_i8 : vector<64xi8> to vector<128xi4>
  %w_i8 = arith.extui %pk_i4 : vector<128xi4> to vector<128xi8>
  return %w_i8 : vector<128xi8>
}

// Negative: a stray extui i4->i8 NOT fed by a bitcast must remain legal
// (no aievec.unpack synthesised).
// CHECK-LABEL: @extui_i4_to_i8_no_bitcast
func.func @extui_i4_to_i8_no_bitcast(%pk_i4: vector<64xi4>) -> vector<64xi8> {
  // CHECK-NOT: aievec.unpack
  // CHECK: arith.extui
  %w_i8 = arith.extui %pk_i4 : vector<64xi4> to vector<64xi8>
  return %w_i8 : vector<64xi8>
}

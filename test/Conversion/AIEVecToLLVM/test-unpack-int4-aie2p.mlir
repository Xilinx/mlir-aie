//===- test-unpack-int4-aie2p.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm='aie-target=aie2p' | FileCheck %s

// AIE2P int4-packed unpack: input is byte-packed (2 nibbles per byte). The
// op result widens lane count by 2x while keeping i8 element type. Maps to
// the I512/I1024 unpack intrinsic with unsigned-mode (sign=0) because AWQ
// packed weights are unsigned and the signed offset is folded into the
// per-group zero point.

// CHECK-LABEL: @test_unpack_i4_v64
func.func @test_unpack_i4_v64(%arg0: vector<32xi8>) -> vector<64xi8> {
  // CHECK:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:  %[[OUT:.*]] = "xllvm.intr.aie2p.unpack.I512.I8.I4"(%arg0, %[[C0]]) : (vector<32xi8>, i32) -> vector<64xi8>
  %0 = aievec.unpack %arg0 : vector<32xi8>, vector<64xi8>
  // CHECK: return %[[OUT]] : vector<64xi8>
  return %0 : vector<64xi8>
}

// CHECK-LABEL: @test_unpack_i4_v128
func.func @test_unpack_i4_v128(%arg0: vector<64xi8>) -> vector<128xi8> {
  // CHECK:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:  %[[OUT:.*]] = "xllvm.intr.aie2p.unpack.I1024.I8.I4"(%arg0, %[[C0]]) : (vector<64xi8>, i32) -> vector<128xi8>
  %0 = aievec.unpack %arg0 : vector<64xi8>, vector<128xi8>
  // CHECK: return %[[OUT]] : vector<128xi8>
  return %0 : vector<128xi8>
}

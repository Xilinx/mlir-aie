//===- test-upd_small.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
// Test a direct load to vector register that does not actually need an update
module {
  func.func @test(%arg0: memref<4x32x64xi16>) {
    %i = arith.constant 1 : index
    %j = arith.constant 2 : index
    %k = arith.constant 3 : index
    %0 = aievec.upd %arg0[%i, %j, %k] {index = 0 : i8, offset = 0: i32} : memref<4x32x64xi16>, vector<16xi16>
    return
  }
}
// CHECK: [[STRUCT:%.+]] = builtin.unrealized_conversion_cast %arg0 : memref<4x32x64xi16> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK: [[ARITH_I:%.+]] = arith.constant 1 : index
// CHECK: [[I:%.+]] = builtin.unrealized_conversion_cast [[ARITH_I]] : index to i64
// CHECK: [[ARITH_J:%.+]] = arith.constant 2 : index
// CHECK: [[J:%.+]] = builtin.unrealized_conversion_cast [[ARITH_J]] : index to i64
// CHECK: [[ARITH_K:%.+]] = arith.constant 3 : index
// CHECK: [[K:%.+]] = builtin.unrealized_conversion_cast [[ARITH_K]] : index to i64
// CHECK: [[PTR:%.+]] = llvm.extractvalue [[STRUCT]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK: [[I_STRIDE:%.+]] = llvm.mlir.constant(2048 : index) : i64
// CHECK: [[I_OFF:%.+]] = llvm.mul [[I]], [[I_STRIDE]] : i64
// CHECK: [[J_STRIDE:%.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK: [[J_OFF:%.+]] = llvm.mul [[J]], [[J_STRIDE]] : i64
// CHECK: [[IJ_OFF:%.+]] = llvm.add [[I_OFF]], [[J_OFF]] : i64
// CHECK: [[OFF:%.+]] = llvm.add [[IJ_OFF]], [[K]] : i64
// CHECK: [[EPTR:%.+]] = llvm.getelementptr [[PTR]][[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, i16
// CHECK: [[VPTR:%.+]] = llvm.bitcast [[EPTR]] : !llvm.ptr to !llvm.ptr
// CHECK: {{.*}} = llvm.load [[VPTR]] {alignment = 1 : i64} : !llvm.ptr -> vector<16xi16>

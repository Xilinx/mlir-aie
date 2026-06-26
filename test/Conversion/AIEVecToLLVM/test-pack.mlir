// Copyright (C) 2018-2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --convert-aievec-to-llvm | FileCheck %s
module {
  func.func @test() {
    %0 = llvm.mlir.undef : vector<16xi16>
    %1 = aievec.pack %0 : vector<16xi16>, vector<16xi8>
    return
  }
}
// CHECK: [[UNDEF:%.+]] = llvm.mlir.undef : vector<16xi16>
// CHECK: {{.*}} = llvm.call @llvm.aie.pack.v16int16([[UNDEF]]) : (vector<16xi16>) -> vector<16xi8>

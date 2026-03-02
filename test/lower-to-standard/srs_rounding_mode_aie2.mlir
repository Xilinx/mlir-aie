//===- srs_rounding_mode_aie2.mlir - SRS rounding mode tests ---*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Verify that AIECoreToStandard sets the SRS rounding mode conditionally:
//  - floor (0) for cores with only float SRS (f32→bf16)
//  - positive_inf (9) for cores with integer SRS (i32→i8)
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck --check-prefix=CHECK-FLOAT %s
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=3" %s | FileCheck --check-prefix=CHECK-INT %s

// Float-only SRS core: rounding mode = floor (register 6 = 0)
// CHECK-FLOAT:  func.func @core_0_2
// CHECK-FLOAT:    call @llvm.aie2.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-FLOAT:    %c6_i32 = arith.constant 6 : i32
// CHECK-FLOAT:    %c0_i32{{.*}} = arith.constant 0 : i32
// CHECK-FLOAT:    call @llvm.aie2.set.ctrl.reg(%c6_i32, %c0_i32

// Integer SRS core: rounding mode = positive_inf (register 6 = 9)
// CHECK-INT:  func.func @core_0_3
// CHECK-INT:    call @llvm.aie2.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-INT:    %c6_i32 = arith.constant 6 : i32
// CHECK-INT:    %c9_i32_0 = arith.constant 9 : i32
// CHECK-INT:    call @llvm.aie2.set.ctrl.reg(%c6_i32, %c9_i32_0)

module @test_srs_rounding {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    // Core with only float SRS (f32 -> bf16): should get floor rounding
    %core02 = aie.core(%t02) {
      %c0 = arith.constant 0 : i32
      %v = arith.constant dense<1.0> : vector<16xf32>
      %srs = aievec.srs %v, %c0 : vector<16xf32>, i32, vector<16xbf16>
      aie.end
    }

    // Core with integer SRS (i32 -> i8): should get positive_inf rounding
    %core03 = aie.core(%t03) {
      %c0 = arith.constant 0 : i32
      %v = arith.constant dense<42> : vector<16xi32>
      %srs = aievec.srs %v, %c0 : vector<16xi32>, i32, vector<16xi8>
      aie.end
    }
  }
}

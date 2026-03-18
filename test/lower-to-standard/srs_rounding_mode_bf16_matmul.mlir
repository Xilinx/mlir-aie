//===- srs_rounding_mode_bf16_matmul.mlir - BF16 matmul rounding ----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Additional rounding mode tests for bf16 matmul (issue #2983):
//  - Integer matmul (i8) does NOT trigger conv_even
//  - BF16 matmul + float SRS: conv_even (12) takes precedence over floor (0)
//  - BF16 matmul_aie2p on AIE2P device
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck --check-prefix=CHECK-INT-MATMUL %s
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=3" %s | FileCheck --check-prefix=CHECK-BF16-FLOAT-SRS %s
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=4" %s | FileCheck --check-prefix=CHECK-NO-MATMUL-NO-SRS %s

// Integer matmul (i8 x i8 -> i32): should NOT trigger conv_even.
// No SRS present either, so no ctrl_reg calls at all.
// CHECK-INT-MATMUL:  func.func @core_0_2
// CHECK-INT-MATMUL-NOT:  call @llvm.aie2.set.ctrl.reg

// BF16 matmul + float SRS (f32->bf16): conv_even (12) takes precedence
// over floor (0). This is the most common direct-codegen scenario where
// matmul accumulates in f32 and SRS truncates back to bf16.
// CHECK-BF16-FLOAT-SRS:  func.func @core_0_3
// CHECK-BF16-FLOAT-SRS:    call @llvm.aie2.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-BF16-FLOAT-SRS:    %c6_i32 = arith.constant 6 : i32
// CHECK-BF16-FLOAT-SRS:    %c12_i32 = arith.constant 12 : i32
// CHECK-BF16-FLOAT-SRS:    call @llvm.aie2.set.ctrl.reg(%c6_i32, %c12_i32)

// Core with no SRS and no matmul: no ctrl_reg calls emitted
// CHECK-NO-MATMUL-NO-SRS:  func.func @core_0_4
// CHECK-NO-MATMUL-NO-SRS-NOT:  call @llvm.aie2.set.ctrl.reg

module @test_bf16_matmul_rounding {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)

    // Integer matmul (i8 x i8 -> i32): should NOT trigger conv_even
    %core02 = aie.core(%t02) {
      %lhs = arith.constant dense<1> : vector<4x8xi8>
      %rhs = arith.constant dense<1> : vector<8x8xi8>
      %acc = arith.constant dense<0> : vector<4x8xi32>
      %res = aievec.matmul %lhs, %rhs, %acc :
        vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
      aie.end
    }

    // BF16 matmul + float SRS (the realistic direct-codegen pattern):
    // matmul in f32, then SRS to bf16. Conv_even should take precedence.
    %core03 = aie.core(%t03) {
      %c0 = arith.constant 0 : i32
      %lhs = arith.constant dense<1.0> : vector<4x8xbf16>
      %rhs = arith.constant dense<1.0> : vector<8x4xbf16>
      %acc = arith.constant dense<0.0> : vector<4x4xf32>
      %res = aievec.matmul %lhs, %rhs, %acc :
        vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
      %srs = aievec.srs %res, %c0 : vector<4x4xf32>, i32, vector<4x4xbf16>
      aie.end
    }

    // Core with only plain arithmetic (no SRS, no matmul): no ctrl_reg
    %core04 = aie.core(%t04) {
      %v = arith.constant dense<1> : vector<16xi32>
      %sum = arith.addi %v, %v : vector<16xi32>
      aie.end
    }
  }
}

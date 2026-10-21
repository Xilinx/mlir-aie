//===- srs_rounding_mode_aie2p.mlir - AIE2P rounding mode tests -----------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Verify conv_even rounding mode on AIE2P, for bf16 matmul_aie2p and for
// float-only SRS (f32->bf16).
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck --check-prefix=CHECK-BF16-AIE2P %s
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=3" %s | FileCheck --check-prefix=CHECK-FLOAT-AIE2P %s
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=4" %s | FileCheck --check-prefix=CHECK-MIXED-AIE2P %s

// BF16 matmul_aie2p: crRnd index fixed for AIE2P (1 instead of 6).
// crSat keeps AIE2 index (9) to preserve existing saturation behavior.
// CHECK-BF16-AIE2P:  func.func @core_0_2
// CHECK-BF16-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-BF16-AIE2P:    %c1_i32_0 = arith.constant 1 : i32
// CHECK-BF16-AIE2P:    %c12_i32 = arith.constant 12 : i32
// CHECK-BF16-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c1_i32_0, %c12_i32)

// Float-only SRS: conv_even, matching arith.truncf.
// CHECK-FLOAT-AIE2P:  func.func @core_0_3
// CHECK-FLOAT-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-FLOAT-AIE2P:    %c1_i32_0 = arith.constant 1 : i32
// CHECK-FLOAT-AIE2P:    %c12_i32 = arith.constant 12 : i32
// CHECK-FLOAT-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c1_i32_0, %c12_i32)

// Float and integer SRS on one core: positive_inf (9), as for integer SRS.
// CHECK-MIXED-AIE2P:  func.func @core_0_4
// CHECK-MIXED-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-MIXED-AIE2P:    %c1_i32_0 = arith.constant 1 : i32
// CHECK-MIXED-AIE2P:    %c9_i32_1 = arith.constant 9 : i32
// CHECK-MIXED-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c1_i32_0, %c9_i32_1)

module @test_aie2p_rounding {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)

    // BF16 matmul_aie2p: should get conv_even rounding via aie2p intrinsic
    %core02 = aie.core(%t02) {
      %lhs = arith.constant dense<1.0> : vector<8x8xbf16>
      %rhs = arith.constant dense<1.0> : vector<8x8xbf16>
      %acc = arith.constant dense<0.0> : vector<8x8xf32>
      %res = aievec.matmul_aie2p %lhs, %rhs, %acc :
        vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
      aie.end
    }

    %t03 = aie.tile(0, 3)

    // Float-only SRS (f32 -> bf16)
    %core03 = aie.core(%t03) {
      %c0 = arith.constant 0 : i32
      %v = arith.constant dense<1.0> : vector<16xf32>
      %srs = aievec.srs %v, %c0 : vector<16xf32>, i32, vector<16xbf16>
      aie.end
    }

    %t04 = aie.tile(0, 4)

    // Float and integer SRS
    %core04 = aie.core(%t04) {
      %c0 = arith.constant 0 : i32
      %f = arith.constant dense<1.0> : vector<16xf32>
      %fs = aievec.srs %f, %c0 : vector<16xf32>, i32, vector<16xbf16>
      %i = arith.constant dense<42> : vector<16xi32>
      %is = aievec.srs %i, %c0 : vector<16xi32>, i32, vector<16xi8>
      aie.end
    }
  }
}

//===- srs_rounding_mode_aie2p.mlir - AIE2P rounding mode tests -----------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Verify conv_even rounding for bf16 matmul_aie2p and for a float-only SRS
// core on AIE2P. The float-only case changed from floor; it is checked
// separately because it is the half extrapolated from npu1 measurement, so
// it is the one that should fail if npu2 ever contradicts it.
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck --check-prefix=CHECK-BF16-AIE2P %s
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=3" %s | FileCheck --check-prefix=CHECK-FLOAT-AIE2P %s

// BF16 matmul_aie2p: crRnd index fixed for AIE2P (1 instead of 6).
// crSat keeps AIE2 index (9) to preserve existing saturation behavior.
// CHECK-BF16-AIE2P:  func.func @core_0_2
// CHECK-BF16-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-BF16-AIE2P:    %c1_i32_0 = arith.constant 1 : i32
// CHECK-BF16-AIE2P:    %c12_i32 = arith.constant 12 : i32
// CHECK-BF16-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c1_i32_0, %c12_i32)

// Float-only SRS on AIE2P: conv_even (register 1 = 12), same as AIE2.
// Constants are captured, not named: the core has its own constant for the
// SRS shift so suffixing is ordering-dependent, and capturing also pins that
// the value passed is the one just checked to be 12.
// CHECK-FLOAT-AIE2P:  func.func @core_0_3
// CHECK-FLOAT-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-FLOAT-AIE2P:    %[[CRRND:.*]] = arith.constant 1 : i32
// CHECK-FLOAT-AIE2P:    %[[CONVEVEN:.*]] = arith.constant 12 : i32
// CHECK-FLOAT-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%[[CRRND]], %[[CONVEVEN]]) : (i32, i32) -> ()

module @test_aie2p_rounding {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    // BF16 matmul_aie2p: should get conv_even rounding via aie2p intrinsic
    %core02 = aie.core(%t02) {
      %lhs = arith.constant dense<1.0> : vector<8x8xbf16>
      %rhs = arith.constant dense<1.0> : vector<8x8xbf16>
      %acc = arith.constant dense<0.0> : vector<8x8xf32>
      %res = aievec.matmul_aie2p %lhs, %rhs, %acc :
        vector<8x8xbf16>, vector<8x8xbf16> into vector<8x8xf32>
      aie.end
    }

    // Float-only SRS (f32 -> bf16) with no matmul: conv_even, as on AIE2.
    %core03 = aie.core(%t03) {
      %c0 = arith.constant 0 : i32
      %v = arith.constant dense<1.0> : vector<16xf32>
      %srs = aievec.srs %v, %c0 : vector<16xf32>, i32, vector<16xbf16>
      aie.end
    }
  }
}

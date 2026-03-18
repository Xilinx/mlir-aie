//===- srs_rounding_mode_aie2p.mlir - AIE2P rounding mode tests -----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Verify conv_even rounding mode for bf16 matmul_aie2p on AIE2P device.
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck --check-prefix=CHECK-BF16-AIE2P %s

// BF16 matmul_aie2p: rounding mode = conv_even (register 6 = 12)
// Uses llvm.aie2p.set.ctrl.reg instead of llvm.aie2.set.ctrl.reg
// CHECK-BF16-AIE2P:  func.func @core_0_2
// CHECK-BF16-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c9_i32, %c1_i32)
// CHECK-BF16-AIE2P:    %c6_i32 = arith.constant 6 : i32
// CHECK-BF16-AIE2P:    %c12_i32 = arith.constant 12 : i32
// CHECK-BF16-AIE2P:    call @llvm.aie2p.set.ctrl.reg(%c6_i32, %c12_i32)

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
  }
}

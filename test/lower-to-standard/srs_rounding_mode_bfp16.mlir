//===- srs_rounding_mode_bfp16.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core with bf16 matmuls rounds to nearest even (conv_even, 12), whether
// it converts their bf16 operands to bfp16 or loads them as bfp16 blocks.

// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=2" %s | FileCheck --check-prefix=CONVERT %s
// RUN: aie-opt --aie-standard-lowering="tilecol=0 tilerow=3" %s | FileCheck --check-prefix=LOADED %s

// CONVERT: func.func @core_0_2
// CONVERT: call @llvm.aie2p.set.ctrl.reg(%{{.*}}, %{{.*}})
// CONVERT: %[[RND:.*]] = arith.constant 1 : i32
// CONVERT: %[[CONV_EVEN:.*]] = arith.constant 12 : i32
// CONVERT: call @llvm.aie2p.set.ctrl.reg(%[[RND]], %[[CONV_EVEN]])

// LOADED: func.func @core_0_3
// LOADED: call @llvm.aie2p.set.ctrl.reg(%{{.*}}, %{{.*}})
// LOADED: %[[RND:.*]] = arith.constant 1 : i32
// LOADED: %[[CONV_EVEN:.*]] = arith.constant 12 : i32
// LOADED: call @llvm.aie2p.set.ctrl.reg(%[[RND]], %[[CONV_EVEN]])

module {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %b03 = aie.buffer(%t03) {sym_name = "blocks"} : memref<144xi8>
    %core02 = aie.core(%t02) {
      %a = arith.constant dense<1.0> : vector<8x8xbf16>
      %m, %e = aievec.to_bfp16 %a : vector<8x8xbf16> -> vector<64xi8>, vector<8xi8>
      aie.end
    }
    %core03 = aie.core(%t03) {
      %c0 = arith.constant 0 : index
      %c72 = arith.constant 72 : index
      %acc = arith.constant dense<0.0> : vector<8x8xf32>
      %am, %ae = aievec.load_bfp16 %b03[%c0] : memref<144xi8> -> vector<64xi8>, vector<8xi8>
      %bm, %be = aievec.load_bfp16 %b03[%c72] : memref<144xi8> -> vector<64xi8>, vector<8xi8>
      %r = aievec.matmul_aie2p_bfp16 %am, %ae, %bm, %be, %acc : vector<64xi8>, vector<8xi8>, vector<64xi8>, vector<8xi8> into vector<8x8xf32>
      aie.end
    }
  }
}

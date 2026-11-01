//===- test-shuffle-aie2p-only-modes.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s --convert-aievec-to-llvm='aie-target=aie2p' | FileCheck %s

// The AIE2P-only modes pass their numbers, 48 to 55, to the vshuffle
// intrinsic. One-input modes repeat lhs as rhs.

// CHECK-LABEL: @aie2p_modes
// CHECK-SAME: %[[I32:[^:]*]]: vector<16xi32>
// CHECK-DAG: %[[M48:.*]] = llvm.mlir.constant(48 : i32) : i32
// CHECK-DAG: %[[M49:.*]] = llvm.mlir.constant(49 : i32) : i32
// CHECK-DAG: %[[M50:.*]] = llvm.mlir.constant(50 : i32) : i32
// CHECK-DAG: %[[M51:.*]] = llvm.mlir.constant(51 : i32) : i32
// CHECK-DAG: %[[M52:.*]] = llvm.mlir.constant(52 : i32) : i32
// CHECK-DAG: %[[M53:.*]] = llvm.mlir.constant(53 : i32) : i32
// CHECK-DAG: %[[M54:.*]] = llvm.mlir.constant(54 : i32) : i32
// CHECK-DAG: %[[M55:.*]] = llvm.mlir.constant(55 : i32) : i32
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%[[I32]], %[[I32]], %[[M48]])
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M49]])
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M50]])
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M51]])
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M52]])
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M53]])
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M54]])
// CHECK-DAG: "xllvm.intr.aie2p.vshuffle"(%{{.*}}, %{{.*}}, %[[M55]])
func.func @aie2p_modes(%i32 : vector<16xi32>, %i64 : vector<8xi64>,
                       %bf16 : vector<32xbf16>, %i8 : vector<64xi8>)
    -> (vector<16xi32>, vector<16xi32>, vector<8xi64>, vector<8xi64>,
        vector<32xbf16>, vector<32xbf16>, vector<16xi32>, vector<64xi8>) {
  %0 = aievec.shuffle %i32 [t32_8x2] : vector<16xi32>
  %1 = aievec.shuffle %i32 [t32_2x8] : vector<16xi32>
  %2 = aievec.shuffle %i64 [t64_4x2] : vector<8xi64>
  %3 = aievec.shuffle %i64 [t64_2x4] : vector<8xi64>
  %4 = aievec.shuffle %bf16, %bf16 [t16_8x8_lo] : vector<32xbf16>
  %5 = aievec.shuffle %bf16, %bf16 [t16_8x8_hi] : vector<32xbf16>
  %6 = aievec.shuffle %i32 [t32_1x16_flip] : vector<16xi32>
  %7 = aievec.shuffle %i8 [t8_1x4_flip] : vector<64xi8>
  return %0, %1, %2, %3, %4, %5, %6, %7 : vector<16xi32>, vector<16xi32>,
      vector<8xi64>, vector<8xi64>, vector<32xbf16>, vector<32xbf16>,
      vector<16xi32>, vector<64xi8>
}

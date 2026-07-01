//===- npu2_instgen_multiple_devices.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that only the selected device and runtime sequence gets configured.

// RUN: aie-translate --aie-npu-to-binary --aie-output-binary=false --aie-device-name=device_b --aie-sequence-name=sequence_b %s | FileCheck %s
module {
  aie.device(npu2) @device_a {
    aie.runtime_sequence @sequence_a(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      %cst_npu_0 = arith.constant 0x01010101 : i32
      %cst_npu_1 = arith.constant 0x01010101 : i32
      aiex.npu.write32(%cst_npu_0, %cst_npu_1) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
    aie.runtime_sequence @sequence_b(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      %cst_npu_2 = arith.constant 0x02020202 : i32
      %cst_npu_3 = arith.constant 0x02020202 : i32
      aiex.npu.write32(%cst_npu_2, %cst_npu_3) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
  }

  aie.device(npu2) @device_b {
    aie.runtime_sequence @sequence_a(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      %cst_npu_4 = arith.constant 0x03030303 : i32
      %cst_npu_5 = arith.constant 0x03030303 : i32
      aiex.npu.write32(%cst_npu_4, %cst_npu_5) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
    aie.runtime_sequence @sequence_b(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      // CHECK: 06040100
      // CHECK: 00000108
      // CHECK: 00000001
      // CHECK: 00000028
      // CHECK: 00000000
      // CHECK: 00000000
      // CHECK: 04240404
      // CHECK: 00000000
      // CHECK: 04040404
      // CHECK: 00000018
      %cst_npu_6 = arith.constant 0x04040404 : i32
      %cst_npu_7 = arith.constant 0x04040404 : i32
      aiex.npu.write32(%cst_npu_6, %cst_npu_7) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
  }
}

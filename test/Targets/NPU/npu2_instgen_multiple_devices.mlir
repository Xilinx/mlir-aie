//===- npu2_instgen_multiple_devices.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// Test that only the selected device and runtime sequence gets configured.

// RUN: aie-translate --aie-npu-to-binary --aie-output-binary=false --aie-device-name=device_b --aie-sequence-name=sequence_b %s | FileCheck %s
module {
  aie.device(npu2) @device_a {
    aie.runtime_sequence @sequence_a(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      %w32_addr = arith.constant 16843009 : i32
      %w32_val = arith.constant 16843009 : i32
      aiex.npu.write32(%w32_addr, %w32_val) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
    aie.runtime_sequence @sequence_b(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      %w32_addr_1 = arith.constant 33686018 : i32
      %w32_val_1 = arith.constant 33686018 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
  }

  aie.device(npu2) @device_b {
    aie.runtime_sequence @sequence_a(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      %w32_addr_2 = arith.constant 50529027 : i32
      %w32_val_2 = arith.constant 50529027 : i32
      aiex.npu.write32(%w32_addr_2, %w32_val_2) {column = 2 : i32, row = 2 : i32} : i32, i32
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
      %w32_addr_3 = arith.constant 67372036 : i32
      %w32_val_3 = arith.constant 67372036 : i32
      aiex.npu.write32(%w32_addr_3, %w32_val_3) {column = 2 : i32, row = 2 : i32} : i32, i32
    }
  }
}

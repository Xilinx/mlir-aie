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
    aiex.runtime_sequence @sequence_a(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      aiex.npu.write32 { column = 2 : i32, row = 2 : i32, address = 0x01010101 : ui32, value = 0x01010101 : ui32 }
    }
    aiex.runtime_sequence @sequence_b(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      aiex.npu.write32 { column = 2 : i32, row = 2 : i32, address = 0x02020202 : ui32, value = 0x02020202 : ui32 }
    }
  }

  aie.device(npu2) @device_b {
    aiex.runtime_sequence @sequence_a(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
      aiex.npu.write32 { column = 2 : i32, row = 2 : i32, address = 0x03030303 : ui32, value = 0x03030303 : ui32 }
    }
    aiex.runtime_sequence @sequence_b(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
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
      aiex.npu.write32 { column = 2 : i32, row = 2 : i32, address = 0x04040404 : ui32, value = 0x04040404 : ui32 }
    }
  }
}

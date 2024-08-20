//===- roundtrip.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// CHECK: aie.device
// CHECK: aiex.npu.dma_wait {symbol = @out0}
aie.device(npu1_4col) {
  memref.global "public" @out0 : memref<16xi32>
  aiex.runtime_sequence() {
    aiex.npu.dma_wait {symbol = @out0}
  }
}

// -----

// CHECK: aie.device
// CHECK: aiex.npu.dma_wait {symbol = @out0}
aie.device(npu1_4col) {
  memref.global "public" @out0 : memref<16xi32>
  aiex.runtime_sequence() {
    aiex.npu.dma_wait {symbol = @out0}
  }
}

// -----

// CHECK: aie.device
// CHECK: aiex.npu.address_patch {addr = 123 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
aie.device(npu1_4col) {
  aiex.runtime_sequence() {
    aiex.npu.address_patch {addr = 123 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
  }
}

// -----

// CHECK: aie.device
// CHECK: runtime_sequence @seq(%arg0: memref<1xi32>)
// CHECK: aiex.npu.write32 {address = 432 : ui32, value = 1 : ui32}
aie.device(npu1_4col) {
  aiex.runtime_sequence @seq(%arg0 : memref<1xi32>) {
    aiex.npu.write32 {address = 432 : ui32, value = 1 : ui32}
  }
}
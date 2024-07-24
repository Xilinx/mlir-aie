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

// CHECK-LABEL: aiex.runtime_sequence
// CHECK: aiex.npu.dma_wait {symbol = @out0}
aie.device(npu1_4col) {
  memref.global "public" @out0 : memref<16xi32>
  aiex.runtime_sequence() {
    aiex.npu.dma_wait {symbol = @out0}
  }
}

// -----

// CHECK-LABEL: aiex.runtime_sequence
// CHECK: aiex.npu.dma_wait {symbol = @out0}
aiex.runtime_sequence() {
  aiex.npu.dma_wait {symbol = @out0}
}

// -----

// CHECK-LABEL: aiex.runtime_sequence
// CHECK: aiex.npu.address_patch {addr = 123 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
aiex.runtime_sequence() {
  aiex.npu.address_patch {addr = 123 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
}
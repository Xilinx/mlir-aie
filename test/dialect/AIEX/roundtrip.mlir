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

// CHECK-LABEL: func.func @npu_dma_wait
// CHECK: aiex.npu.dma_wait {symbol = @out0}
aie.device(npu) {
  memref.global "public" @out0 : memref<16xi32>
  func.func @npu_dma_wait() {
    aiex.npu.dma_wait {symbol = @out0}
    return
  }
}

// -----

// CHECK-LABEL: func.func @npu_dma_wait_no_device
// CHECK: aiex.npu.dma_wait {symbol = @out0}
func.func @npu_dma_wait_no_device() {
  aiex.npu.dma_wait {symbol = @out0}
  return
}

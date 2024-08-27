//===- roundtrip_npu1_4col.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-instgen -aie-output-binary=true %s -o ./cfg.bin
// RUN: %python txn2mlir.py -f ./cfg.bin | FileCheck %s

// CHECK: aie.device(npu1_4col)
// CHECK: aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
// CHECK: aiex.npu.maskwrite32 {address = 35856384 : ui32, mask = 1 : ui32, value = 1 : ui32}
// CHECK: aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 1 : ui32, value = 1 : ui32}
// CHECK: aiex.npu.maskwrite32 {address = 102965248 : ui32, mask = 1 : ui32, value = 1 : ui32}
module {
  aie.device(npu1_4col) {
    aiex.runtime_sequence() {
      aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 35856384 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 1 : ui32, value = 1 : ui32}
      aiex.npu.maskwrite32 {address = 102965248 : ui32, mask = 1 : ui32, value = 1 : ui32}
    }
  }
}

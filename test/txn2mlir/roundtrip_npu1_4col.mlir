//===- roundtrip_npu1.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-to-binary %s -o ./roundtrip_npu1_cfg.bin
// RUN: %python txn2mlir.py -f ./roundtrip_npu1_cfg.bin | FileCheck %s

// CHECK: aie.device(npu1)
// CHECK: aiex.npu.maskwrite32 {address = 2301952 : ui32, mask = 1 : ui32, value = 1 : ui32}
// CHECK: aiex.npu.maskwrite32 {address = 35856384 : ui32, mask = 1 : ui32, value = 1 : ui32}
// CHECK: aiex.npu.maskwrite32 {address = 69410816 : ui32, mask = 1 : ui32, value = 1 : ui32}
// CHECK: aiex.npu.maskwrite32 {address = 102965248 : ui32, mask = 1 : ui32, value = 1 : ui32}
module {
  aie.device(npu1) {
    aie.runtime_sequence() {
      %cst_npu_0 = arith.constant 2301952 : i32
      %cst_npu_1 = arith.constant 1 : i32
      %cst_npu_2 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%cst_npu_0, %cst_npu_1, %cst_npu_2) : i32, i32, i32
      %cst_npu_3 = arith.constant 35856384 : i32
      %cst_npu_4 = arith.constant 1 : i32
      %cst_npu_5 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32
      %cst_npu_6 = arith.constant 69410816 : i32
      %cst_npu_7 = arith.constant 1 : i32
      %cst_npu_8 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%cst_npu_6, %cst_npu_7, %cst_npu_8) : i32, i32, i32
      %cst_npu_9 = arith.constant 102965248 : i32
      %cst_npu_10 = arith.constant 1 : i32
      %cst_npu_11 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%cst_npu_9, %cst_npu_10, %cst_npu_11) : i32, i32, i32
    }
  }
}

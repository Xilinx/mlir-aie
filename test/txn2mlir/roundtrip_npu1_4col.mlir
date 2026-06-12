//===- roundtrip_npu1.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-to-binary %s -o ./roundtrip_npu1_cfg.bin
// RUN: %python txn2mlir.py -f ./roundtrip_npu1_cfg.bin | FileCheck %s

// CHECK: aie.device(npu1)
// CHECK-DAG: %[[M0:.+]] = arith.constant 2301952 : i32
// CHECK: aiex.npu.maskwrite32(%[[M0]],
// CHECK-DAG: %[[M1:.+]] = arith.constant 35856384 : i32
// CHECK: aiex.npu.maskwrite32(%[[M1]],
// CHECK-DAG: %[[M2:.+]] = arith.constant 69410816 : i32
// CHECK: aiex.npu.maskwrite32(%[[M2]],
// CHECK-DAG: %[[M3:.+]] = arith.constant 102965248 : i32
// CHECK: aiex.npu.maskwrite32(%[[M3]],
module {
  aie.device(npu1) {
    aie.runtime_sequence() {
      %mw_addr = arith.constant 2301952 : i32
      %mw_val = arith.constant 1 : i32
      %mw_mask = arith.constant 1 : i32
      aiex.npu.maskwrite32(%mw_addr, %mw_val, %mw_mask) : i32, i32, i32
      %mw_addr_1 = arith.constant 35856384 : i32
      %mw_val_1 = arith.constant 1 : i32
      %mw_mask_1 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%mw_addr_1, %mw_val_1, %mw_mask_1) : i32, i32, i32
      %mw_addr_2 = arith.constant 69410816 : i32
      %mw_val_2 = arith.constant 1 : i32
      %mw_mask_2 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%mw_addr_2, %mw_val_2, %mw_mask_2) : i32, i32, i32
      %mw_addr_3 = arith.constant 102965248 : i32
      %mw_val_3 = arith.constant 1 : i32
      %mw_mask_3 = arith.constant 1 : i32
      aiex.npu.maskwrite32(%mw_addr_3, %mw_val_3, %mw_mask_3) : i32, i32, i32
    }
  }
}

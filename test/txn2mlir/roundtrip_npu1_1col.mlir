//===- roundtrip_npu1_1col.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-to-binary %s -o ./roundtrip_npu1_1col_cfg.bin
// RUN: %python txn2mlir.py -f ./roundtrip_npu1_1col_cfg.bin | FileCheck %s

// CHECK: aie.device(npu1_1col)
// CHECK: memref.global "private" constant @config_blockwrite_data_0 : memref<2xi32> = dense<[4195328, 0]>
// CHECK-DAG: %[[MWA:.+]] = arith.constant 2301952 : i32
// CHECK: aiex.npu.maskwrite32(%[[MWA]],
// CHECK-DAG: %[[WA0:.+]] = arith.constant 2224128 : i32
// CHECK-DAG: %[[WV0:.+]] = arith.constant 2 : i32
// CHECK: aiex.npu.write32(%[[WA0]], %[[WV0]])
// CHECK: aiex.npu.blockwrite(%0) {address = 2215936 : ui32} : memref<2xi32>
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @blockwrite_data : memref<2xi32> = dense<[4195328, 0]>
    aie.runtime_sequence() {
      %mw_addr = arith.constant 2301952 : i32
      %mw_val = arith.constant 2 : i32
      %mw_mask = arith.constant 2 : i32
      aiex.npu.maskwrite32(%mw_addr, %mw_val, %mw_mask) : i32, i32, i32
      %w32_addr_1 = arith.constant 2224128 : i32
      %w32_val_1 = arith.constant 2 : i32
      aiex.npu.write32(%w32_addr_1, %w32_val_1) : i32, i32
      %0 = memref.get_global @blockwrite_data : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 2215936 : ui32} : memref<2xi32>
    }
  }
}

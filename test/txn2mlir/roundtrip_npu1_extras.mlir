//===- roundtrip_npu1_extras.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-to-binary %s -o %t.cfg
// RUN: %python txn2mlir.py -f %t.cfg | FileCheck %s

// CHECK: aie.device(npu1_1col)
// CHECK: memref.global "private" constant @config_blockwrite_data
// sync(column, row, direction, channel, column_num, row_num) = (0, 0, 1, 2, 1, 1)
// CHECK-DAG: %[[COL:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[ROW:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[DIR:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[CHAN:.+]] = arith.constant 2 : i32
// CHECK-DAG: %[[CNUM:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[RNUM:.+]] = arith.constant 1 : i32
// CHECK: aiex.npu.sync(%[[COL]], %[[ROW]], %[[DIR]], %[[CHAN]], %[[CNUM]], %[[RNUM]])
// CHECK: aiex.npu.load_pdi {address = 305419896 : ui64, id = 7 : i32, size = 4096 : i32}
// CHECK-DAG: %[[AP0:.+]] = arith.constant 4 : i32
// CHECK: aiex.npu.address_patch(%[[AP0]] : i32) {addr = 123456 : ui32, arg_idx = 3 : i32}
// CHECK: aiex.npu.preempt {level = 2 : ui8}
// CHECK-DAG: %[[WA0:.+]] = arith.constant 2224128 : i32
// CHECK-DAG: %[[WV0:.+]] = arith.constant 2 : i32
// CHECK: aiex.npu.write32(%[[WA0]], %[[WV0]])
// CHECK: aiex.npu.blockwrite
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @blockwrite_data : memref<2xi32> = dense<[1, 2]>
    aie.runtime_sequence() {
      %col = arith.constant 0 : i32
      %row = arith.constant 0 : i32
      %dir = arith.constant 1 : i32
      %chan = arith.constant 2 : i32
      %col_num = arith.constant 1 : i32
      %row_num = arith.constant 1 : i32
      aiex.npu.sync(%col, %row, %dir, %chan, %col_num, %row_num) : i32, i32, i32, i32, i32, i32
      aiex.npu.load_pdi {id = 7 : i32, size = 4096 : i32, address = 305419896 : ui64}
      %arg_plus = arith.constant 4 : i32
      aiex.npu.address_patch(%arg_plus : i32) {addr = 123456 : ui32, arg_idx = 3 : i32}
      aiex.npu.preempt {level = 2 : ui8}
      %w32_addr = arith.constant 2224128 : i32
      %w32_val = arith.constant 2 : i32
      aiex.npu.write32(%w32_addr, %w32_val) : i32, i32
      %0 = memref.get_global @blockwrite_data : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 2215936 : ui32} : memref<2xi32>
    }
  }
}

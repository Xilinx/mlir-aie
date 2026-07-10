//===- roundtrip_npu1_extras.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate -aie-npu-to-binary %s -o %t.cfg
// RUN: %python txn2mlir.py -f %t.cfg | FileCheck %s

// CHECK: aie.device(npu1_1col)
// CHECK: memref.global "private" constant @config_blockwrite_data
// CHECK: aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
// CHECK: aiex.npu.load_pdi {address = 305419896 : ui64, id = 7 : i32, size = 4096 : i32}
// CHECK: %[[AP:.*]] = arith.constant 4 : i32
// CHECK: aiex.npu.address_patch(%[[AP]] : i32) {addr = 123456 : ui32, arg_idx = 3 : i32}
// CHECK: aiex.npu.preempt {level = 2 : ui8}
// CHECK-DAG: %[[W_VAL:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[W_ADDR:.*]] = arith.constant 2224128 : i32
// CHECK: aiex.npu.write32(%[[W_ADDR]], %[[W_VAL]]) : i32, i32
// CHECK: aiex.npu.blockwrite
module {
  aie.device(npu1_1col) {
    memref.global "private" constant @blockwrite_data : memref<2xi32> = dense<[1, 2]>
    aie.runtime_sequence() {
      %cst_npu_0 = arith.constant 0 : i32
      %cst_npu_1 = arith.constant 0 : i32
      %cst_npu_2 = arith.constant 1 : i32
      %cst_npu_3 = arith.constant 2 : i32
      %cst_npu_4 = arith.constant 1 : i32
      %cst_npu_5 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_0, %cst_npu_1, %cst_npu_2, %cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32, i32, i32, i32
      aiex.npu.load_pdi {id = 7 : i32, size = 4096 : i32, address = 305419896 : ui64}
      %cst_npu_6 = arith.constant 4 : i32
      aiex.npu.address_patch(%cst_npu_6 : i32) {addr = 123456 : ui32, arg_idx = 3 : i32}
      aiex.npu.preempt {level = 2 : ui8}
      %cst_npu_7 = arith.constant 2224128 : i32
      %cst_npu_8 = arith.constant 2 : i32
      aiex.npu.write32(%cst_npu_7, %cst_npu_8) : i32, i32
      %0 = memref.get_global @blockwrite_data : memref<2xi32>
      aiex.npu.blockwrite(%0) {address = 2215936 : ui32} : memref<2xi32>
    }
  }
}

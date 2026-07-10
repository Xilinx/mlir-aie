//===- npu_instgen.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-to-binary -aie-output-binary=false %s | FileCheck %s
module {
  aie.device(npu1) {
    memref.global "private" constant @write_data : memref<8xi32> = dense<[100, 101, 102, 103, 104 ,105, 106, 107]>
    aie.runtime_sequence(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {

      // TXN header 0.1
      // CHECK: 06030100
      // CHECK: 00000104
      // CHECK: 00000007
      // CHECK: 000000DC

      // CHECK: 00000000
      // CHECK: 00000000
      // CHECK: 06400DEF
      // CHECK: 00000000
      // CHECK: 00000042
      // CHECK: 00000018
      %cst_npu_0 = arith.constant 0xabc00def : i32
      %cst_npu_1 = arith.constant 0x42 : i32
      aiex.npu.write32(%cst_npu_0, %cst_npu_1) {column = 3 : i32, row = 4 : i32} : i32, i32

      // CHECK: 00000000
      // CHECK: 00000000
      // CHECK: ABC00DEF
      // CHECK: 00000000
      // CHECK: 00000314
      // CHECK: 00000018
      %cst_npu_2 = arith.constant 0xabc00def : i32
      %cst_npu_3 = arith.constant 0x314 : i32
      aiex.npu.write32(%cst_npu_2, %cst_npu_3) : i32, i32

      // CHECK: 00000001
      // CHECK: 00000000
      // CHECK: 12345679
      // CHECK: 00000030
      // CHECK: 00000064
      // CHECK: 00000065
      // CHECK: 00000066
      // CHECK: 00000067
      // CHECK: 00000068
      // CHECK: 00000069
      // CHECK: 0000006A
      // CHECK: 0000006B
      %0 = memref.get_global @write_data : memref<8xi32>
      aiex.npu.blockwrite (%0) {address = 0x12345679 : ui32} : memref<8xi32>

      // CHECK: 00000001
      // CHECK: 00000101
      // CHECK: 02100064
      // CHECK: 00000030
      // CHECK: 00000064
      // CHECK: 00000065
      // CHECK: 00000066
      // CHECK: 00000067
      // CHECK: 00000068
      // CHECK: 00000069
      // CHECK: 0000006A
      // CHECK: 0000006B
      aiex.npu.blockwrite (%0) { column = 1 : i32, row = 1 : i32, address = 100 : ui32} : memref<8xi32>

      // CHECK: 00000003
      // CHECK: 00000000
      // CHECK: 0430567A
      // CHECK: 00000000
      // CHECK: 00001001
      // CHECK: F00FF00F
      // CHECK: 0000001C
      %cst_npu_4 = arith.constant 0x0000567A : i32
      %cst_npu_5 = arith.constant 0x1001 : i32
      %cst_npu_6 = arith.constant 0xf00ff00f : i32
      aiex.npu.maskwrite32(%cst_npu_4, %cst_npu_5, %cst_npu_6) {column = 2 : i32, row = 3 : i32} : i32, i32, i32

      // CHECK: 00000080
      // CHECK: 00000010
      // CHECK: 00030401
      // CHECK: 05010200
      %cst_npu_7 = arith.constant 3 : i32
      %cst_npu_8 = arith.constant 4 : i32
      %cst_npu_9 = arith.constant 1 : i32
      %cst_npu_10 = arith.constant 5 : i32
      %cst_npu_11 = arith.constant 1 : i32
      %cst_npu_12 = arith.constant 2 : i32
      aiex.npu.sync(%cst_npu_7, %cst_npu_8, %cst_npu_9, %cst_npu_10, %cst_npu_11, %cst_npu_12) : i32, i32, i32, i32, i32, i32

      // CHECK: 00020008
      // CHECK: 00000400
      // CHECK: 12345678
      // CHECK: 00000ABC
      aiex.npu.load_pdi { address = 0xabc12345678 : ui64, id = 2 : i32, size = 1024 : i32 }
    }
  }
}

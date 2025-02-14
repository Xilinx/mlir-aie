//===- npu_instgen.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-npu-instgen %s | FileCheck %s
module {
  aie.device(npu1) {
    memref.global "private" constant @write_data : memref<8xi32> = dense<[100, 101, 102, 103, 104 ,105, 106, 107]>
    aiex.runtime_sequence(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {

      // TXN header 1.0
      // CHECK: 06030001
      // CHECK: 00000105
      // CHECK: 00000006
      // CHECK: 000000A0

      // CHECK: 00000000
      // CHECK: 06400DEF
      // CHECK: 00000042
      aiex.npu.write32 { column = 3 : i32, row = 4 : i32, address = 0xabc00def : ui32, value = 0x42 : ui32 }

      // CHECK: 00000000
      // CHECK: ABC00DEF
      // CHECK: 00000314
      aiex.npu.write32 { address = 0xabc00def : ui32, value = 0x314 : ui32 }

      // CHECK: 00000001
      // CHECK: 12345679
      // CHECK: 0000002C
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
      // CHECK: 02100064
      // CHECK: 0000002C
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
      // CHECK: 0430567A
      // CHECK: 00001001
      // CHECK: F00FF00F
      aiex.npu.maskwrite32 { column = 2 : i32, row = 3 : i32, address = 0x0000567A : ui32, value = 0x1001 : ui32, mask = 0xf00ff00f : ui32 }

      // CHECK: 00000080
      // CHECK: 00000010
      // CHECK: 00030401
      // CHECK: 05010200
      aiex.npu.sync { column = 3 : i32, row = 4 : i32, direction = 1 : i32, channel = 5 : i32, column_num = 1 : i32, row_num = 2 : i32 }
    }
  }
}

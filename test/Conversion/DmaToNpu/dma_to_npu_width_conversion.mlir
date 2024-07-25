//===- dma_to_npu_width_conversion.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 3rd 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s 2>&1 | FileCheck %s

// CHECK-LABEL:  aie.device(xcve2302) {
// CHECK:  memref.global "public" @toMem : memref<65536xbf16>
// CHECK:  memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[8192, 0, 0, 33554432, -2080374657, 31, 0, 33554432]>
// CHECK:  func.func @sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<65536xbf16>) {
// CHECK:    %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
// CHECK:    aiex.npu.blockwrite(%0) {address = 118784 : ui32} : memref<8xi32>
// CHECK:    aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}
// CHECK:    aiex.npu.write32 {address = 119300 : ui32, column = 0 : i32, row = 0 : i32, value = 2147680256 : ui32}
// CHECK:    aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK:    }
// CHECK:    aie.shim_dma_allocation @toMem(S2MM, 0, 0)

module @shimDmaMemcpy{
  aie.device(xcve2302) {
    memref.global "public" @toMem : memref<65536xbf16>
    func.func @sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<65536xbf16>) {
      aiex.npu.dma_memcpy_nd (2, 0, %arg0[0, 0, 0, 0][4, 4, 64, 64][0, 64, 256, 1]) {id = 0 : i64, metadata = @toMem} : memref<65536xbf16>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}


//===- dma_to_ipu.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-dma-to-ipu %s | FileCheck %s

// TODO - more
// CHECK-LABEL: test0
// CHECK: aiex.ipu.writebd_shimtile
// CHECK-SAME: ddr_id = 0 : i32
// CHECK-SAME: valid_bd = 1 : i32
// CHECK: aiex.ipu.writebd_shimtile
// CHECK-SAME: ddr_id = 1 : i32
module  {
aie.device(ipu) {
  memref.global "public" @toMem : memref<16xi32>
  memref.global "public" @fromMem : memref<16xi32>
  func.func @test0(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    aiex.ipu.dma_memcpy_nd(0, 0, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64][%c1_i64, %c1_i64, %c16_i64, %c16_i64][%c0_i64, %c0_i64, %c64_i64]) { metadata = @toMem, id = 1 : i64 } : memref<16xi32>
    aiex.ipu.dma_memcpy_nd(0, 1, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64][%c1_i64, %c1_i64, %c16_i64, %c16_i64][%c0_i64, %c0_i64, %c64_i64]) { metadata = @fromMem, id = 0 : i64 } : memref<16xi32>
    return
  }
  aie.shim_dma_allocation @fromMem (MM2S, 0, 0)
  aie.shim_dma_allocation @toMem (S2MM, 0, 0)
}
}

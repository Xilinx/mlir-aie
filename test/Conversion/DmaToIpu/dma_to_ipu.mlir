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
// CHECK: AIEX.ipu.writebd_shimtile
// CHECK-SAME: ddr_id = 0 : i32
// CHECK-SAME: valid_bd = 1 : i32
// CHECK: AIEX.ipu.writebd_shimtile
// CHECK-SAME: ddr_id = 1 : i32
module  {
AIE.device(ipu) {
  memref.global "public" @toMem : memref<16xi32>
  memref.global "public" @fromMem : memref<16xi32>
  func.func @test0(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    AIEX.ipu.dma_memcpy_nd(%c0_i32, %c0_i32, %arg0[%c0_i32, %c0_i32, %c0_i32, %c0_i32][%c1_i32, %c1_i32, %c16_i32, %c16_i32][%c0_i32, %c0_i32, %c64_i32]) { metadata = @toMem, id = 1 : i32 } : (i32, i32, memref<16xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
    AIEX.ipu.dma_memcpy_nd(%c0_i32, %c1_i32, %arg1[%c0_i32, %c0_i32, %c0_i32, %c16_i32][%c1_i32, %c1_i32, %c16_i32, %c16_i32][%c0_i32, %c0_i32, %c64_i32]) { metadata = @fromMem, id = 0 : i32 } : (i32, i32, memref<16xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
    return
  }
  AIE.shim_dma_allocation @fromMem (MM2S, 0, 0)
  AIE.shim_dma_allocation @toMem (S2MM, 0, 0)
}
}

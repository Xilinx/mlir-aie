//===- npu_dma_memcpy.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// CHECK: memref.global "private" constant {{.*}} : memref<8xi32> = dense<[2048, 0, 0, 33554432, -2113929153, 33556479, 0, 33554432]>
// CHECK: aiex.runtime_sequence
// CHECK:   %[[VALUE:.*]] = memref.get_global {{.*}} : memref<8xi32>
// CHECK:   aiex.npu.blockwrite(%[[VALUE]]) {address = 118784 : ui32} : memref<8xi32>
// CHECK:   aiex.npu.address_patch {addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 16384 : i32}
// CHECK:   aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
// CHECK: }

module {
 aie.device(npu1_1col) {
  %t00 = aie.tile(0, 0)
  aie.shim_dma_allocation @airMemcpyId12(MM2S, 0, 0)
  memref.global "public" @airMemcpyId12 : memref<1x2x1x32x32xi32, 1 : i32>
  aiex.runtime_sequence (%arg0: memref<2x64x64xi32>, %arg1: memref<2x64x64xi32>, %arg2: memref<2x64x64xi32>) {
    aiex.npu.dma_memcpy_nd(%arg0[1, 0, 0, 0][1, 2, 32, 32][4096, 2048, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId12} : memref<2x64x64xi32>
  }
 }
}

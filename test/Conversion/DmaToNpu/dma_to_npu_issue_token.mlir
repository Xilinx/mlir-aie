//===- dma_to_npu_issue_token.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-dma-to-npu %s | FileCheck %s

// TODO - more
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK: aiex.npu.write32
// CHECK-SAME: value = 2147483649
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 1 : i32
// CHECK: aiex.npu.write32
// CHECK-SAME: value = 0
module  {
  aie.device(npu1_4col) {
    memref.global "public" @toMem : memref<16xi32>
    memref.global "public" @fromMem : memref<16xi32>
    aiex.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
        aiex.npu.dma_memcpy_nd (0, 0, %arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64, issue_token = true } : memref<16xi32>
        aiex.npu.dma_memcpy_nd (0, 1, %arg1[0, 0, 0, 16][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @fromMem, id = 0 : i64, issue_token = false } : memref<16xi32>
    }
    aie.shim_dma_allocation @fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @toMem (S2MM, 0, 0)
  }
}

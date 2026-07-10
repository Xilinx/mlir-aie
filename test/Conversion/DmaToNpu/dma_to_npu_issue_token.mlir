//===- dma_to_npu_issue_token.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-dma-to-npu %s | FileCheck %s

// TODO - more
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK: %[[V0:.*]] = arith.constant -2147483647 : i32
// CHECK: aiex.npu.write32(%{{.*}}, %[[V0]]) : i32, i32
// CHECK: aiex.npu.blockwrite
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 1 : i32
// CHECK: %[[A1:.*]] = arith.constant 119316 : i32
// CHECK: aiex.npu.write32(%[[A1]], %{{.*}}) : i32, i32
module  {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
        aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @toMem, id = 1 : i64, issue_token = true } : memref<16xi32>
        aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 16][1, 1, 16, 16][0, 0, 64, 1]) { metadata = @fromMem, id = 0 : i64, issue_token = false } : memref<16xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @fromMem (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}

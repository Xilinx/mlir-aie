//===- islinear-strides-check.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression test: isLinearTransfer must check strides as well as sizes.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize --aie-dma-tasks-to-npu --split-input-file %s | FileCheck %s



// -----

// Test 1: sizes=[1,1,8] with non-zero outer strides; canonicalization zeroes
// size-1 strides so this becomes a contiguous (linear) transfer.
// CHECK-LABEL: aiex.npu.writebd
// CHECK-SAME:  d0_size = 0 : i32
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8 sizes = [1, 1, 8] strides = [4, 4, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}



// -----

// Test 2: single linear dimension [<size=8,stride=1>] -> linear path (d0_size=0).
// CHECK-LABEL: aiex.npu.writebd
// CHECK-SAME:  d0_size = 0 : i32
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %c0_i32 = arith.constant 0 : i32
      %c8_i32 = arith.constant 8 : i32
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8 sizes = [8] strides = [1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

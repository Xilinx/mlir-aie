//===- islinear-strides-check.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression test: isLinearTransfer must check strides as well as sizes.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu --split-input-file %s | FileCheck %s

// -----

// Test 1: sizes=[1,1,8] with non-zero outer strides -> ND path (d0_size=8).
// CHECK-LABEL: aiex.npu.writebd
// CHECK-SAME:  d0_size = 8 : i32
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32>, 0, 8,
          [<size = 1, stride = 4>, <size = 1, stride = 4>, <size = 8, stride = 1>]
        ) {bd_id = 0 : i32}
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
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32>, 0, 8,
          [<size = 8, stride = 1>]
        ) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

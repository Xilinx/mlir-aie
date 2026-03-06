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
// Regression test for the isLinearTransfer predicate in AIEDMATasksToNPU.cpp.
//
// The predicate must check strides in addition to sizes.  A BD where outer
// dimensions have size=1 but non-zero strides was previously misclassified as
// linear (the old predicate only checked sizes[1]==1 && sizes[2]==1).  The
// corrected predicate additionally requires strides[0]==1 and
// strides[1]==strides[2]==0, matching isLinearTransferWithoutTransformation()
// used by NpuDmaMemcpyNdOp.
//
// Test 1: dimensions with sizes=[1,1,8] and non-zero outer strides.
//   outermost-first: [<size=1,stride=4>, <size=1,stride=4>, <size=8,stride=1>]
//   innermost-first: sizes=[8,1,1], strides=[1,4,4]
//   Old predicate (sizes only): sizes[1]==1 && sizes[2]==1 -> isLinear=true
//     -> d0_size=0 (flat path, d0 wrap limit check skipped — incorrect)
//   New predicate (sizes+strides): strides[1]=4!=0 -> isLinear=false
//     -> d0_size=8 (ND path — correct)
//
// Test 2: single truly-linear dimension [<size=8,stride=1>].
//   innermost-first: sizes=[8,1,1], strides=[1,0,0]
//   New predicate: isLinear=true -> d0_size=0 (flat path — correct)
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu --split-input-file %s | FileCheck %s

// -----

// Test 1: sizes=[1,1,8] with non-zero outer strides -> ND path (d0_size=8).
// CHECK: d0_size = 8 : i32
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
// CHECK: d0_size = 0 : i32
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

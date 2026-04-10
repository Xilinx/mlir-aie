//===- canonicalize_linear_dma_task.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests that the NpuDmaMemcpyNdOp linearization canonicalization fires
// correctly when the op appears alongside DMA task ops
// (aiex.dma_configure_task / aiex.dma_start_task / aiex.dma_await_task).
//
// The DMA task infrastructure (aiex.dma_configure_task, aie.dma_bd) uses a
// different op set from aiex.npu.dma_memcpy_nd, so the LinearizeContiguous-
// Transfer pattern does not apply to it.  These tests confirm that
// aiex.npu.dma_memcpy_nd is linearized while aiex.dma_configure_task ops are
// left untouched.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s

// -----

// A contiguous aiex.npu.dma_memcpy_nd next to a aiex.dma_configure_task.
// Only the dma_memcpy_nd should be folded; the dma_configure_task body is
// unchanged.

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @mixed_task_and_memcpy_nd
// -- The dma_memcpy_nd should be folded to linear form.
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// -- The dma_configure_task op itself must still be present.
// CHECK:           aiex.dma_configure_task
// CHECK:           aiex.dma_start_task
// CHECK:           aiex.dma_await_task
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence @mixed_task_and_memcpy_nd(%arg0 : memref<2x512xi32>) {
      // This op is contiguous and should be linearised by --canonicalize.
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<2x512xi32>

      // This DMA task op uses a different op (aie.dma_bd) and is not touched
      // by the NpuDmaMemcpyNdOp canonicalization pattern.
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<2x512xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
    aie.shim_dma_allocation @of_fromMem (%tile_0_0, MM2S, 0)
  }
}

// -----

// A contiguous aiex.npu.dma_memcpy_nd with issue_token in the DMA task style.
// sizes=[1,1,2,512] strides=[0,0,512,1] -> sizes=[1,1,1,1024] strides=[0,0,0,1]

// CHECK-LABEL: aie.device(npu1)
// CHECK:         aie.runtime_sequence @dma_task_style_fold
// CHECK:           aiex.npu.dma_memcpy_nd
// CHECK-SAME:        [0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]
// CHECK-SAME:        issue_token = true
module {
  aie.device(npu1) {
    aie.runtime_sequence @dma_task_style_fold(%arg0 : memref<2x512xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 2, 512][0, 0, 512, 1])
        { metadata = @of_fromMem, id = 0 : i64, issue_token = true } : memref<2x512xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

//===- decompose_large_dma_task_iteration.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An oversized task-path aie.dma_bd that carries the iteration attribute cannot
// be decomposed.

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @iteration_oversized(%in: memref<8192xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        // expected-error @+1 {{buffer descriptor with the iteration attribute is too large to lower and cannot be decomposed}}
        aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 2062 sizes = [1, 1031, 2] strides = [0, 3, 1])
          {iteration = #aie.bd_iteration<size = 4, stride = 2062, current = 0>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

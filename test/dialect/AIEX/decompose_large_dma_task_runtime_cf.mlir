//===- decompose_large_dma_task_runtime_cf.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-lower-dynamic-bd-pool,aie-dma-tasks-to-npu))' \
// RUN:   %s | FileCheck %s

// The 1031-long dimension of decompose_large_dma_task.mlir's SLICE case, inside
// a runtime-bound loop. It splits into the same 2-BD chain, the dynamic BD pool
// gives each BD its own id, and the first BD's next_bd is the second's id.

// CHECK-LABEL: @slice_in_loop
// CHECK:       scf.for
// CHECK:         %[[BD0:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:         %[[BD1:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:         arith.andi %[[BD1]]
// CHECK:         aiex.npu.blockwrite_values
// CHECK:         aiex.npu.blockwrite_values
// CHECK:         aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %[[BD0]]
// CHECK:         aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[BD0]] : i32
// CHECK:         aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[BD1]] : i32
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @slice_in_loop(%in: memref<4096xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %n step %c1 {
        %tk = aiex.dma_configure_task_for @a {
          aie.dma_bd(%in : memref<4096xi32> offset = 0 len = 2062 sizes = [1, 1, 1031, 2] strides = [0, 0, 3, 1])
            {burst_length = 0 : i32}
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%tk)
        aiex.dma_await_task(%tk)
        aiex.dma_free_task(%tk)
      }
    }
  }
}

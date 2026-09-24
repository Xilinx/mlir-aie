//===- dma_task_iteration_attr.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// End-to-end lowering tests for #aie.bd_iteration on the dma_configure_task_for
// path (shim allocation + BD-ID assignment + tasks-to-npu). Checks that:
//   - iteration_size  is stored as (attr.size - 1)
//   - iteration_stride is stored as (attr.stride * elem_bytes / 4 - 1), i.e.
//     word-scaled then minus-one biased
//   - when no iteration attr is present both fields are zero
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' --split-input-file %s | FileCheck %s

// -----

// Test 1: Explicit iteration with i32 elements.
// iteration_size  = size - 1 = 4 - 1 = 3
// iteration_stride = stride * sizeof(i32) / 4 - 1 = 1024 * 4 / 4 - 1 = 1023
// repeat_count = 3 → push_queue repeat operand is 3

// CHECK-LABEL: @iter_i32
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_current = 0
// CHECK-SAME: iteration_size = 3
// CHECK-SAME: iteration_stride = 1023
// CHECK: aiex.npu.push_queue
// CHECK-SAME: repeat %{{.*}}
module @iter_i32 {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @seq(%buf: memref<8192xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%buf : memref<8192xi32> offset = 0 len = 2048
                   sizes = [1, 1, 2048] strides = [0, 0, 1])
          {iteration = #aie.bd_iteration<size = 4, stride = 1024, current = 0>}
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Test 2: No iteration attribute (pure repeat_count).
// Without an iteration attr, both iteration_size and iteration_stride must be 0.

// CHECK-LABEL: @pure_repeat
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_size = 0
// CHECK-SAME: iteration_stride = 0
module @pure_repeat {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @seq(%buf: memref<1024xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 1024
                   sizes = [1, 1, 1024] strides = [0, 0, 1])
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Test 3: Explicit iteration with i16 elements.
// iteration_size  = size - 1 = 4 - 1 = 3
// iteration_stride = stride * sizeof(i16) / 4 - 1 = 512 * 2 / 4 - 1 = 255
// (i16 is 2 bytes wide; the hardware stride field counts 32-bit words)

// CHECK-LABEL: @iter_i16
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_size = 3
// CHECK-SAME: iteration_stride = 255
module @iter_i16 {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @seq(%buf: memref<8192xi16>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%buf : memref<8192xi16> offset = 0 len = 2048
                   sizes = [1, 1, 2048] strides = [0, 0, 1])
          {iteration = #aie.bd_iteration<size = 4, stride = 512, current = 0>}
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

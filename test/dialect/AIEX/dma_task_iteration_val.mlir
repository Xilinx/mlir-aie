//===- dma_task_iteration_val.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Companion to dma_task_iteration_attr.mlir for the runtime iteration operands
// (iteration_size_val / iteration_stride_val).  When those operands fold to
// compile-time constants -- e.g. a dynamic design specialized to constant
// M/N -- the BD must take the static writebd path and encode iteration exactly
// like #aie.bd_iteration, NOT emit aiex.npu.blockwrite_values (which only the
// C++ TXN target can translate).  This is the lowering half of the regression
// for the static-specialization path of npu-xrt/matmul_whole_array_dynamic.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' --split-input-file %s | FileCheck %s

// -----

// Test 1: Constant iteration operands with i32 elements. Must match the
// #aie.bd_iteration<size = 4, stride = 1024> encoding in dma_task_iteration_attr.mlir:
//   iteration_size  = size - 1 = 4 - 1 = 3
//   iteration_stride = stride * sizeof(i32) / 4 - 1 = 1024 * 4 / 4 - 1 = 1023

// CHECK-LABEL: @iter_val_i32
// CHECK-NOT: aiex.npu.blockwrite_values
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_current = 0
// CHECK-SAME: iteration_size = 3
// CHECK-SAME: iteration_stride = 1023
module @iter_val_i32 {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @seq(%buf: memref<8192xi32>) {
      %size = arith.constant 4 : i32
      %stride = arith.constant 1024 : i32
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%buf : memref<8192xi32> offset = 0 len = 2048
                   sizes = [1, 1, 2048] strides = [0, 0, 1])
          iteration_size_val %size : i32 iteration_stride_val %stride : i32
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Test 2: Constant iteration operands with i16 elements. Must match the
// #aie.bd_iteration<size = 4, stride = 512> encoding (i16 is 2 bytes wide):
//   iteration_size  = 4 - 1 = 3
//   iteration_stride = 512 * 2 / 4 - 1 = 255

// CHECK-LABEL: @iter_val_i16
// CHECK-NOT: aiex.npu.blockwrite_values
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_size = 3
// CHECK-SAME: iteration_stride = 255
module @iter_val_i16 {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @seq(%buf: memref<8192xi16>) {
      %size = arith.constant 4 : i32
      %stride = arith.constant 512 : i32
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%buf : memref<8192xi16> offset = 0 len = 2048
                   sizes = [1, 1, 2048] strides = [0, 0, 1])
          iteration_size_val %size : i32 iteration_stride_val %stride : i32
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

// -----

// Test 3: Constant iteration size with a zero iteration stride is a pure
// repeat: the count is carried by the task's repeat_count queue push, so the
// BD iteration fields stay zero (and it must still take the static path).

// CHECK-LABEL: @iter_val_pure_repeat
// CHECK-NOT: aiex.npu.blockwrite_values
// CHECK: aiex.npu.writebd
// CHECK-SAME: iteration_size = 0
// CHECK-SAME: iteration_stride = 0
module @iter_val_pure_repeat {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @seq(%buf: memref<8192xi32>) {
      %size = arith.constant 8 : i32
      %stride = arith.constant 0 : i32
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%buf : memref<8192xi32> offset = 0 len = 2048
                   sizes = [1, 1, 2048] strides = [0, 0, 1])
          iteration_size_val %size : i32 iteration_stride_val %stride : i32
        aie.end
      } {issue_token = true, repeat_count = 7 : i32}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

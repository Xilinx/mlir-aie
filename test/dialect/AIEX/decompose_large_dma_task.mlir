//===- decompose_large_dma_task.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for aie-decompose-large-dma-bd on task-path aie.dma_bd ops inside
// aiex.dma_configure_task_for regions (IRON rt.fill/drain tap lowering).
//
//===----------------------------------------------------------------------===//


// -----

// Test 1: FACTOR — oversized non-contiguous shim BD is rewritten in place to a
// single hardware-legal aie.dma_bd (no next_bd chain).
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=FACTOR

// FACTOR-LABEL: @factor_task_bd
// FACTOR:         aiex.dma_configure_task_for @a
// FACTOR:           aie.dma_bd
// FACTOR-NOT:       aie.next_bd
// FACTOR-NOT:       4, 1920]
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @factor_task_bd(%in: memref<7684xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<7684xi32> offset = 0 len = 7680 sizes = [1, 1, 4, 1920] strides = [0, 0, 1921, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}


// -----

// Test 2: UNCHANGED — a small already-legal task BD is left as-is.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=UNCHANGED

// UNCHANGED-LABEL: @small_unchanged_task
// UNCHANGED:         aie.dma_bd
// UNCHANGED-SAME:        sizes = [1, 1, 1, 8]
// UNCHANGED-SAME:        strides = [0, 0, 0, 1]
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @small_unchanged_task(%in: memref<8xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<8xi32> offset = 0 len = 8 sizes = [1, 1, 1, 8] strides = [0, 0, 0, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}


// -----

// Test 3: LOWER — end-to-end through BD-ID assignment and tasks-to-npu.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-substitute-shim-dma-allocations,aie-decompose-large-dma-bd,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER

// LOWER-LABEL: @lower_task_bd
// LOWER-NOT:     exceeds the [0:1023] range
// LOWER:         aiex.npu.writebd
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a (%t, MM2S, 0)
    aie.runtime_sequence @lower_task_bd(%in: memref<7684xi32>) {
      %tk = aiex.dma_configure_task_for @a {
        aie.dma_bd(%in : memref<7684xi32> offset = 0 len = 7680 sizes = [1, 1, 4, 1920] strides = [0, 0, 1921, 1])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tk)
      aiex.dma_await_task(%tk)
    }
  }
}

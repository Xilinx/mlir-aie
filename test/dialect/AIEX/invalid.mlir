//===- invalid.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  aie.runtime_sequence() {
    // expected-error@+1 {{'aiex.npu.dma_wait' op couldn't find symbol in parent device}}
    aiex.npu.dma_wait {symbol = @out0}
  }
}

// -----

// An await identifies its channel EITHER by the task operand OR by the full
// sync_* attribute set, never both.
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<1024xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    // expected-error@+1 {{must not set the sync_* channel attributes when a task operand is present}}
    aiex.dma_await_task(%t) {sync_channel = 0 : i32, sync_col = 0 : i32, sync_direction = 1 : i32, sync_row = 0 : i32}
  }
}

// -----

// The sync_* attributes come as a group: all four or none.
aie.device(npu1) {
  aie.runtime_sequence() {
    // expected-error@+1 {{sync_col, sync_row, sync_direction and sync_channel must all be set together or all omitted}}
    aiex.dma_await_task() {sync_col = 0 : i32, sync_row = 0 : i32}
  }
}

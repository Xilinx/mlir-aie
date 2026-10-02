// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics \
// RUN:   --split-input-file %s | FileCheck %s

// An await of %task first consumes the older raw push's token. Its descriptor
// must remain reserved until a second await consumes the task's own token.
// CHECK-LABEL: @raw_token
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @raw_token(%buf: memref<256xi32>) {
    %c0 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32
    %raw = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 7 : i32}
      aie.end
    }
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c7 repeat %c0 {issue_token = true} : i32, i32
    aiex.dma_start_task(%task)
    aiex.dma_await_task(%task)
    %pending = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_await_task(%task)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// Explicit MM2S memcpy tokens participate in the same FIFO.
// CHECK-LABEL: @memcpy_token
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, MM2S, 0)
  aie.runtime_sequence @memcpy_token(%buf: memref<256xi32>) {
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 7 : i64, metadata = @alloc, issue_token = true} : memref<256xi32>
    aiex.dma_start_task(%task)
    aiex.dma_await_task(%task)
    %pending = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_await_task(%task)
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// S2MM memcpy lowering issues a token even when its issue_token is false.
// CHECK-LABEL: @implicit_memcpy_token
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, S2MM, 0)
  aie.runtime_sequence @implicit_memcpy_token(%buf: memref<256xi32>) {
    %task = aiex.dma_configure_task(%tile, S2MM, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 7 : i64, metadata = @alloc, issue_token = false} : memref<256xi32>
    aiex.dma_start_task(%task)
    aiex.dma_await_task(%task)
    %pending = aiex.dma_configure_task(%tile, S2MM, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_await_task(%task)
    %reuse = aiex.dma_configure_task(%tile, S2MM, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids --verify-diagnostics %s \
// RUN:   | FileCheck %s

// A memcpy wait consumes the raw token before the task starts. That token must
// not remain in the allocator's FIFO and delay the task's later release.
// CHECK-LABEL: @memcpy_wait
// CHECK: aiex.npu.dma_wait
// CHECK: aiex.dma_await_task
// CHECK: aie.dma_bd({{.*}} {bd_id = 0 : i32}
// A memcpy wait can also finish a task whose release was deferred by an await
// that named the task but consumed an older token.
// CHECK: aiex.dma_await_task
// CHECK: aiex.npu.dma_wait
// CHECK: aie.dma_bd({{.*}} {bd_id = 1 : i32}
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, MM2S, 0)
  aie.runtime_sequence @memcpy_wait(%buf: memref<256xi32>) {
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) {id = 7 : i64, metadata = @alloc, issue_token = true} : memref<256xi32>
    aiex.npu.dma_wait {symbol = @alloc}
    aiex.dma_start_task(%a)
    aiex.dma_await_task(%a)
    %b = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.end
    } {issue_token = true}
    %c = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%b)
    aiex.dma_start_task(%c)
    aiex.dma_await_task(%c)
    aiex.npu.dma_wait {symbol = @alloc}
    %reuse = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256) {bd_id = 1 : i32}
      aie.end
    }
  }
}

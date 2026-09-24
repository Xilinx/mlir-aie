//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=true' %s \
// RUN:   | FileCheck %s
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=true' \
// RUN:   --aie-dma-tasks-to-npu %s | FileCheck %s --check-prefix=PUSH

// A queue push carries at most 255 repeats (256 runs of the chain). A start
// asking for more is issued as several starts of the same task: full 256-run
// ones first, then the remainder. Only the last keeps the token, so an await on
// the task returns after every run. Each is its own queue push, so the fifth
// on a 4-deep channel gets a queue-space poll like any other push.

// A count the field holds is left alone.
// CHECK-LABEL: @at_max
// CHECK:       aiex.dma_start_task(%{{.*}})
// CHECK-NOT:   no_token
// CHECK-NOT:   aiex.dma_start_task
// PUSH-LABEL:  @at_max
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c255_i32{{(_[0-9]+)?}} {issue_token = true}
// PUSH-NOT:    aiex.npu.push_queue

// 601 runs = 256 + 256 + 89.
// CHECK-LABEL: @split_600
// CHECK:       aiex.dma_start_task(%[[T:.*]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {repeat_count = 88 : i32}
// CHECK-NEXT:  aiex.dma_await_task(%[[T]])
// PUSH-LABEL:  @split_600
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c255_i32{{(_[0-9]+)?}} {issue_token = false}
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c255_i32{{(_[0-9]+)?}} {issue_token = false}
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c88_i32{{(_[0-9]+)?}} {issue_token = true}
// PUSH:        aiex.npu.sync
// PUSH-NOT:    aiex.npu.push_queue

// 512 runs split evenly; the remainder is a full chunk too.
// CHECK-LABEL: @split_511
// CHECK:       aiex.dma_start_task(%[[T:.*]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {repeat_count = 255 : i32}
// CHECK-NOT:   aiex.dma_start_task
// PUSH-LABEL:  @split_511

// The per-start override replaces the task's count for that start only, and is
// split like any other count. A start that already withholds its token keeps
// withholding it on the last chunk.
// CHECK-LABEL: @override
// CHECK:       aiex.dma_start_task(%[[T:.*]]) {repeat_count = 3 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {no_token, repeat_count = 44 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]])
// CHECK-NEXT:  aiex.dma_await_task(%[[T]])
// PUSH-LABEL:  @override
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c3_i32{{(_[0-9]+)?}} {issue_token = true}
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c255_i32{{(_[0-9]+)?}} {issue_token = false}
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c44_i32{{(_[0-9]+)?}} {issue_token = false}
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c7_i32{{(_[0-9]+)?}} {issue_token = true}

// 1280 runs is five full pushes. The fifth lands on a full 4-deep queue, so
// it waits for a free slot (bit 22 of shim MM2S_0's status, 0x1D228).
// CHECK-LABEL: @fifth_slot
// CHECK-COUNT-4: aiex.dma_start_task(%{{.*}}) {no_token, repeat_count = 255 : i32}
// CHECK-DAG:   arith.constant 119336 : i32
// CHECK-DAG:   arith.constant 4194304 : i32
// CHECK:       aiex.npu.maskpoll
// CHECK-NEXT:  aiex.dma_start_task(%{{.*}}) {repeat_count = 255 : i32}
// CHECK-NOT:   aiex.npu.maskpoll
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @at_max(%arg0: memref<256xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true, repeat_count = 255 : i32}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
  }
  aie.runtime_sequence @split_600(%arg0: memref<256xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true, repeat_count = 600 : i32}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
  }
  aie.runtime_sequence @split_511(%arg0: memref<256xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {repeat_count = 511 : i32}
    aiex.dma_start_task(%t)
  }
  aie.runtime_sequence @override(%arg0: memref<256xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true, repeat_count = 7 : i32}
    aiex.dma_start_task(%t) {repeat_count = 3 : i32}
    aiex.dma_start_task(%t) {repeat_count = 300 : i32, no_token}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_await_task(%t)
  }
  aie.runtime_sequence @fifth_slot(%arg0: memref<256xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {repeat_count = 1279 : i32}
    aiex.dma_start_task(%t)
  }
}

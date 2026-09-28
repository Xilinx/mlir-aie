// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=false' \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu='enforce-queue-depth=false' \
// RUN:   --verify-diagnostics %s | FileCheck %s --check-prefix=CLEAN \
// RUN:   --implicit-check-not=aiex.dma_queue_overflow_diagnosed --implicit-check-not=aiex.npu.maskpoll
// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=false' \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=GUARD

// A prior warning must neither suppress another channel/sequence's warning nor
// prevent the final combined pass from guarding the already-diagnosed channel.
// CLEAN: @first
// CLEAN: @second
// GUARD-LABEL: @first
// GUARD-COUNT-2: aiex.npu.maskpoll
// GUARD-NOT: aiex.npu.maskpoll
// GUARD-LABEL: @second
// GUARD: aiex.npu.maskpoll
// GUARD-NOT: aiex.npu.maskpoll
aie.device(npu2) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @first(%arg: memref<256xi32>) {
    %c0 = arith.constant 0 : i32
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%arg : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    // expected-warning@+1 {{whose task queue is only 4 deep}}
    aiex.dma_start_task(%task)
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
    // expected-warning@+1 {{whose task queue is only 4 deep}}
    aiex.npu.push_queue (0, 0, MM2S:1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
  aie.runtime_sequence @second(%arg: memref<256xi32>) {
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%arg : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    // expected-warning@+1 {{whose task queue is only 4 deep}}
    aiex.dma_start_task(%task)
  }
}

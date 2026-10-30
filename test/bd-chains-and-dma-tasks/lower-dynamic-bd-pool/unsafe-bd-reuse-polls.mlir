// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: sed '/expected-warning/d' %s | aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=false' \
// RUN:   --verify-diagnostics --split-input-file

// The poll before the fifth push leaves up to three queued tasks AND one
// running task. The first, multi-BD task may therefore still be running.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @running_after_space_poll(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 128)
        aie.next_bd ^second
      ^second:
        aie.dma_bd(%buf : memref<256xi32> offset = 128 len = 128)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// Full channel idleness proves even a non-token task complete.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @safe_idle_poll(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %status = arith.constant 119336 : i32
    %idle = arith.constant 7864380 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.maskpoll(%status, %c0, %idle) : i32, i32, i32
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// Idleness leaves the older token pending; an await cannot credit a new task.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @idle_preserves_tokens(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %status = arith.constant 119336 : i32
    %idle = arith.constant 7864380 : i32
    scf.if %cond {
      %old = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%old)
      aiex.npu.maskpoll(%status, %c0, %idle) : i32, i32, i32
      aiex.dma_free_task(%old)
      %new = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%new)
      aiex.dma_await_task(%new)
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%new)
    }
  }
}

// -----

// Wrong-channel, partial-mask and nonzero-idle-bit polls do not prove idleness.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @not_idle_polls(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %running = arith.constant 4 : i32
    %status = arith.constant 119336 : i32
    %other_status = arith.constant 119328 : i32
    %idle = arith.constant 7864380 : i32
    %partial = arith.constant 7864376 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.maskpoll(%other_status, %c0, %idle) : i32, i32, i32
      aiex.npu.maskpoll(%status, %c0, %partial) : i32, i32, i32
      aiex.npu.maskpoll(%status, %running, %idle) : i32, i32, i32
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// The next real poll proves that at least one of those five pushes completed.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @completed_by_later_poll(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.dma_free_task(%task)
    }
  }
}

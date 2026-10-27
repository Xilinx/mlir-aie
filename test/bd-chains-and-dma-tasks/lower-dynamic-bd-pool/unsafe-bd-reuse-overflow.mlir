// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true enforce-queue-depth=false' \
// RUN:   --verify-diagnostics --split-input-file %s

// Abstract queue truncation is not hardware progress. Retain the old owner
// even after more than depth+1 pushes, all without completion tokens.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @no_poll_no_completion(%buf: memref<256xi32>, %cond: i1) {
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
      // expected-warning@+1 {{whose task queue is only 4 deep}}
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// A fifth push may be dropped. A subsequent space poll can succeed with the
// first task still running; attempted pushes do not prove FIFO progress.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @dropped_push_then_poll(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %status = arith.constant 119336 : i32
    %space = arith.constant 4194304 : i32
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
      // expected-warning@+1 {{whose task queue is only 4 deep}}
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.maskpoll(%status, %c0, %space) : i32, i32, i32
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// Polls must also preserve uncertainty for an owner already abstracted out.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @forgotten_owner_then_poll(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %status = arith.constant 119336 : i32
    %space = arith.constant 4194304 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      // expected-warning@+1 {{whose task queue is only 4 deep}}
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.maskpoll(%status, %c0, %space) : i32, i32, i32
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// A later known token still proves an abstractly truncated non-token prefix
// complete. Truncation must retain uncertainty without making it permanent.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @later_token_completes_prefix(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      // expected-warning@+1 {{whose task queue is only 4 deep}}
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_free_task(%task)
    }
  }
}

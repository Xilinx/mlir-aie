// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' \
// RUN:   --verify-diagnostics --split-input-file %s

// The task await consumes the older raw push's token, not its own.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @older_raw_token(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    scf.if %cond {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// An explicit MM2S memcpy token also precedes the task's token.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, MM2S, 0)
  aie.runtime_sequence @older_memcpy_token(%buf: memref<256xi32>, %cond: i1) {
    scf.if %cond {
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1])
        {id = 7 : i64, metadata = @alloc, issue_token = true} : memref<256xi32>
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// S2MM memcpy always issues a token, even with issue_token=false.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, S2MM, 0)
  aie.runtime_sequence @implicit_memcpy_token(%buf: memref<256xi32>, %cond: i1) {
    scf.if %cond {
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1])
        {id = 7 : i64, metadata = @alloc, issue_token = false} : memref<256xi32>
      %task = aiex.dma_configure_task(%tile, S2MM, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
      // expected-warning@+1 {{with no completion guarantee}}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// A raw sync retires the task token on the same physical channel.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @safe_raw_sync(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// A metadata-based wait consumes the same channel token as a task await.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.shim_dma_allocation @alloc (%tile, MM2S, 0)
  aie.runtime_sequence @safe_dma_wait(%buf: memref<256xi32>, %cond: i1) {
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.npu.dma_wait {symbol = @alloc}
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// A later raw token proves an earlier non-token task complete as well.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @safe_later_raw_token(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.if %cond {
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%task)
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_free_task(%task)
    }
  }
}

// -----

// A different direction's token cannot steal this task's completion.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @independent_direction(%buf: memref<256xi32>, %cond: i1) {
    %c0 = arith.constant 0 : i32
    scf.if %cond {
      aiex.npu.push_queue (0, 0, S2MM:0) bd_id %c0 repeat %c0 {issue_token = true} : i32, i32
      %task = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
      aiex.dma_free_task(%task)
    }
  }
}

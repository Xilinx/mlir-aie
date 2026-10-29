//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' \
// RUN:         --verify-diagnostics --split-input-file %s
// RUN: aie-opt --aie-lower-dynamic-bd-pool \
// RUN:         --verify-diagnostics --split-input-file %s
// RUN: sed '/expected-/d' %s | aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=false' \
// RUN:         --verify-diagnostics --split-input-file

// aiex.dma_free_task returns a buffer descriptor id to the runtime pool with
// no completion guarantee, and the pool is free to hand that id straight back
// out to the very next task it draws. Freeing the previous iteration's input
// task right after starting the next one -- without awaiting it first -- is
// the hazard four in-tree npu-xrt tests shipped with until it was caught by
// intermittent silicon failures (Xilinx/mlir-aie#3974).

// Freed with no completion guarantee at all, then reused by the next
// iteration's start.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @unsafe_free_without_await(%buf: memref<256xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      // expected-warning@+1 {{frees a task on tile (0,0) MM2S channel 0 with no completion guarantee, so its buffer descriptor id goes back to the runtime pool while the DMA may still be running}}
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    // Unbounded unconsumed tokens exceed the finite completion history.
    // The final await cannot be credited to the last task, so this warns too.
    // expected-warning@+1 {{frees a task on tile (0,0) MM2S channel 0 with no completion guarantee, so its buffer descriptor id goes back to the runtime pool while the DMA may still be running}}
    aiex.dma_free_task(%last)
  }
}

// -----

// The fix #3974 applied: await the previous task before freeing it, the same
// idiom dynamic_pingpong_iterarg_await already used. Quiet.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @safe_await_before_free(%buf: memref<256xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%prev)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}

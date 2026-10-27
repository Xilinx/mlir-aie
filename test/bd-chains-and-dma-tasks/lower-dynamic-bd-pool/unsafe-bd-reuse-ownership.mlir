// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: sed '/expected-warning/d' %s | aie-opt --aie-lower-dynamic-bd-pool --split-input-file 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not='with no completion guarantee'
// CHECK: aie.runtime_sequence

// The representative configure is awaited, but the selected else task is not.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @unsafe_if_result(%buf: memref<256xi32>, %cond: i1) {
    %task = scf.if %cond -> (index) {
      %then = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%then)
      aiex.dma_await_task(%then)
      scf.yield %then : index
    } else {
      %else = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%else)
      scf.yield %else : index
    }
    // expected-warning@+1 {{with no completion guarantee}}
    aiex.dma_free_task(%task)
  }
}

// -----

// Awaiting each selected task on its own branch proves the result complete.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @safe_if_result(%buf: memref<256xi32>, %cond: i1) {
    %task = scf.if %cond -> (index) {
      %then = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%then)
      aiex.dma_await_task(%then)
      scf.yield %then : index
    } else {
      %else = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%else)
      aiex.dma_await_task(%else)
      scf.yield %else : index
    }
    aiex.dma_free_task(%task)
  }
}

// -----

// The unstarted init is safe, but each replacement is in flight at loop exit.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @unsafe_loop_result(%buf: memref<256xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %next = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%next)
      scf.yield %next : index
    }
    // expected-warning@+1 {{with no completion guarantee}}
    aiex.dma_free_task(%last)
  }
}

// -----

// A zero-trip loop result still denotes the outstanding initial task.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @zero_trip_result(%buf: memref<256xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c0 to %c0 step %c1 iter_args(%prev = %init) -> (index) {
      scf.yield %prev : index
    }
    // expected-warning@+1 {{with no completion guarantee}}
    aiex.dma_free_task(%last)
  }
}

// -----

// Aliases established before a start must also identify that push.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @start_through_alias(%buf: memref<256xi32>, %cond: i1) {
    %task = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %alias = scf.if %cond -> (index) {
      scf.yield %task : index
    } else {
      scf.yield %task : index
    }
    aiex.dma_start_task(%alias)
    // expected-warning@+1 {{with no completion guarantee}}
    aiex.dma_free_task(%task)
  }
}

// -----

// Simultaneous backedge transfers can swap tasks without merging identities.
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @swapped_iter_args(%buf: memref<256xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %done = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%done)
    aiex.dma_await_task(%done)
    %busy = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%busy)
    %last:2 = scf.for %i = %c0 to %n step %c1
        iter_args(%x = %done, %y = %busy) -> (index, index) {
      scf.yield %y, %x : index, index
    }
    // expected-warning@+1 {{with no completion guarantee}}
    aiex.dma_free_task(%last#0)
    // expected-warning@+1 {{with no completion guarantee}}
    aiex.dma_free_task(%last#1)
  }
}

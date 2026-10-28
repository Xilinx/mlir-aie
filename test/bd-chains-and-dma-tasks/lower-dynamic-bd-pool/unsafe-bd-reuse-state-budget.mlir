// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed '/expected-/d' %s | aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=false' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DISABLED
// RUN: aie-opt --aie-lower-dynamic-bd-pool \
// RUN:   --verify-diagnostics %s | FileCheck %s --check-prefix=GUARD
// RUN: aie-opt --aie-lower-dynamic-bd-pool='warn-unsafe-bd-reuse=true' \
// RUN:   --verify-diagnostics %s | FileCheck %s --check-prefix=GUARD

// Rotation and a transposition generate all 8! permutations. The ownership
// budget must fall back conservatively rather than enumerate them all, without
// disabling queue-depth enforcement in subsequent loops.
// DISABLED-NOT: warning:
// DISABLED: aie.runtime_sequence
// DISABLED-NOT: warning:
// DISABLED: aiex.npu.maskpoll
// DISABLED-NOT: warning:
// GUARD: aiex.npu.maskpoll
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @permuted_completed_tasks(%buf: memref<256xi32>, %n: index, %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %a = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%a)
    aiex.dma_await_task(%a)
    %b = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%b)
    aiex.dma_await_task(%b)
    %c = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%c)
    aiex.dma_await_task(%c)
    %d = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%d)
    aiex.dma_await_task(%d)
    %e = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%e)
    aiex.dma_await_task(%e)
    %f = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%f)
    aiex.dma_await_task(%f)
    %g = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%g)
    aiex.dma_await_task(%g)
    %h = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%h)
    aiex.dma_await_task(%h)
    %last:8 = scf.for %i = %c0 to %n step %c1
        iter_args(%p = %a, %q = %b, %r = %c, %s = %d,
                  %t = %e, %u = %f, %v = %g, %w = %h)
        -> (index, index, index, index, index, index, index, index) {
      %next:8 = scf.if %cond -> (index, index, index, index, index, index, index, index) {
        scf.yield %q, %r, %s, %t, %u, %v, %w, %p : index, index, index, index, index, index, index, index
      } else {
        scf.yield %q, %p, %r, %s, %t, %u, %v, %w : index, index, index, index, index, index, index, index
      }
      scf.yield %next#0, %next#1, %next#2, %next#3, %next#4, %next#5, %next#6, %next#7 : index, index, index, index, index, index, index, index
    }
    // expected-note@+2 {{task ownership analysis budget exhausted}}
    // expected-warning@+1 {{with no completion guarantee}}
    aiex.dma_free_task(%last#0)
    scf.for %i = %c0 to %n step %c1 {
      aiex.npu.push_queue (0, 0, MM2S:0) bd_id %zero repeat %zero {issue_token = false} : i32, i32
    }
  }
}

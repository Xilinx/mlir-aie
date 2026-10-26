//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-opt --aie-lower-dynamic-bd-pool --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-lower-dynamic-bd-pool='enforce-queue-depth=false' \
// RUN:   --verify-diagnostics --split-input-file %s

// A loop iter_arg is not defined by a configure, but every start still occupies
// a queue slot. Repeated starts must be guarded even when the BD is reused.
// CHECK-LABEL: @iter_arg_start
// CHECK: scf.for
// CHECK: aiex.npu.maskpoll
// CHECK-NEXT: aiex.dma_start_task
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @iter_arg_start(%buf: memref<256xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%task = %init) -> (index) {
      // expected-warning@+1 {{whose task queue is only 4 deep}}
      aiex.dma_start_task(%task)
      scf.yield %task : index
    }
  }
}

// -----

// An await through the iter_arg balances a directly defined start. Missing its
// queue effect used to insert an unnecessary poll (or report an overflow).
// CHECK-LABEL: @iter_arg_await
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @iter_arg_await(%buf: memref<256xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%task = %init) -> (index) {
      aiex.dma_start_task(%init)
      aiex.dma_await_task(%task)
      scf.yield %task : index
    }
  }
}

// -----

// Loop results also carry metadata, including in the straight-line suffix.
// CHECK-LABEL: @for_result_start
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_start_task
// CHECK: aiex.npu.maskpoll
// CHECK-NEXT: aiex.dma_start_task
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @for_result_start(%buf: memref<256xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile, MM2S, 0) {
      aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
      aie.end
    }
    %last = scf.for %i = %c0 to %n step %c1 iter_args(%task = %init) -> (index) {
      scf.yield %task : index
    }
    aiex.dma_start_task(%last)
    aiex.dma_start_task(%last)
    aiex.dma_start_task(%last)
    aiex.dma_start_task(%last)
    // expected-warning@+1 {{whose task queue is only 4 deep}}
    aiex.dma_start_task(%last)
  }
}

// -----

// Starts of an if result must be counted on its agreed physical channel.
// CHECK-LABEL: @if_result_start
// CHECK: aiex.npu.maskpoll
// CHECK-NEXT: aiex.dma_start_task
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @if_result_start(%buf: memref<256xi32>, %cond: i1) {
    %task = scf.if %cond -> (index) {
      %then = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      scf.yield %then : index
    } else {
      %else = aiex.dma_configure_task(%tile, MM2S, 0) {
        aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
        aie.end
      }
      scf.yield %else : index
    }
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    aiex.dma_start_task(%task)
    // expected-warning@+1 {{whose task queue is only 4 deep}}
    aiex.dma_start_task(%task)
  }
}

// -----

// Branch-local starts are balanced by the await on their merged result.
// CHECK-LABEL: @if_result_await
// CHECK-NOT: aiex.npu.maskpoll
aie.device(npu1) {
  %tile = aie.tile(0, 0)
  aie.runtime_sequence @if_result_await(%buf: memref<256xi32>, %n: index, %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %task = scf.if %cond -> (index) {
        %then = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%then)
        scf.yield %then : index
      } else {
        %else = aiex.dma_configure_task(%tile, MM2S, 0) {
          aie.dma_bd(%buf : memref<256xi32> offset = 0 len = 256)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%else)
        scf.yield %else : index
      }
      aiex.dma_await_task(%task)
    }
  }
}

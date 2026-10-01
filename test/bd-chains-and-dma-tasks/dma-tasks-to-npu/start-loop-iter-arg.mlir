//===- start-loop-iter-arg.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu --split-input-file --verify-diagnostics %s | FileCheck %s

// A start on a task carried through an scf.for iter_arg lowers when every
// value the iter_arg can hold comes from the same configure: the push names
// that configure's head BD, channel and repeat count.

// CHECK-LABEL: @start_identity_carry
// CHECK: scf.for
// CHECK-DAG: %[[BD:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[RC:.*]] = arith.constant 2 : i32
// CHECK: aiex.npu.push_queue(2, 0, S2MM : 3) bd_id %[[BD]] repeat %[[RC]] {issue_token = true}
// CHECK: aiex.npu.push_queue(2, 0, S2MM : 3) bd_id %{{.*}} repeat %{{.*}} {issue_token = true}
module {
  aie.device(npu1) {
    %tile_2_0 = aie.tile(2, 0)
    aie.runtime_sequence @start_identity_carry(%arg0: memref<1024xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %init = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 5 : i32}
        aie.end
      } {issue_token = true, repeat_count = 2 : i32}
      %last = scf.for %i = %c0 to %n step %c1 iter_args(%t = %init) -> (index) {
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%t)
        scf.yield %t : index
      }
      aiex.dma_start_task(%last)
      aiex.dma_await_task(%last)
    }
  }
}

// -----

// A task reconfigured each iteration can carry either configure's head BD, so
// a single push cannot name it.
module {
  aie.device(npu1) {
    %tile_2_0 = aie.tile(2, 0)
    aie.runtime_sequence @start_two_configures(%arg0: memref<1024xi32>, %n: index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %init = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      %last = scf.for %i = %c0 to %n step %c1 iter_args(%prev = %init) -> (index) {
        // expected-error@+2 {{starts a task carried through control flow that does not come from exactly one aiex.dma_configure_task}}
        // expected-error@+1 {{failed to legalize operation 'aiex.dma_start_task'}}
        aiex.dma_start_task(%prev)
        aiex.dma_await_task(%prev)
        %t = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256) {bd_id = 1 : i32}
          aie.end
        } {issue_token = true}
        scf.yield %t : index
      }
    }
  }
}

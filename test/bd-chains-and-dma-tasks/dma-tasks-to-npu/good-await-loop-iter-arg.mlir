//===- good-await-loop-iter-arg.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A task handle is carried across scf.for iterations via an iter_arg and
// awaited from INSIDE the loop body (on the iter_arg itself, before the next
// iteration's task is configured and yielded) -- the pattern needed for a
// software-pipelined DMA sequence, where the previous iteration's transfer
// must be awaited before its buffer descriptor is reused, not just awaited
// once after the whole loop. `%prev` here is a block argument (the loop's
// own iter_arg), not an op result, so `DMAAwaitTaskOp::getTaskOp()` cannot
// find a defining op (Value::getDefiningOp() returns null for a block
// argument) and must fall back to walking the region-branch predecessors of
// `%prev` (its loop init and the previous iteration's yield) back to the
// configure that fixes its physical channel.

// CHECK-LABEL: @await_loop_iter_arg
// CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %{{.*}} {issue_token = true} : i32, i32
// CHECK: scf.for
// column=0, row=0, direction=1 (MM2S), channel=0, col_num=1, row_num=1
// CHECK: aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
// CHECK: }
// column=0, row=0, direction=1 (MM2S), channel=0, col_num=1, row_num=1
// CHECK: aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence @await_loop_iter_arg(%arg0: memref<1024xi32>, %n: index) {
      %c1 = arith.constant 1 : index
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256) {bd_id = 1 : i32}
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%prev)
        scf.yield %t : index
      }
      aiex.dma_await_task(%last)
    }
  }
}

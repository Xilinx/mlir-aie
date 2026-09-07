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
//
// The loop's own chain (%init, %t) uses tile (2, 0) / S2MM / channel 3 --
// distinct from the unrelated decoy task on tile (0, 0) / MM2S / channel 0 --
// so a resolution bug that wandered onto the decoy, or a channel-index bug,
// produces different, FileCheck-visible values rather than the coincidental
// (0, 0, ...) that two same-default tasks would silently share.

// CHECK-LABEL: @await_loop_iter_arg
// decoy: sync operands: column=0, row=0, direction=1, channel=0, column_num=1, row_num=1
// CHECK: aiex.npu.sync
// CHECK: scf.for
// sync operands: column=2, row=0, direction=0, channel=3, column_num=1, row_num=1
// CHECK-DAG: %[[COL:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CHAN:.*]] = arith.constant 3 : i32
// CHECK: aiex.npu.sync(%[[COL]], %{{.*}}, %{{.*}}, %[[CHAN]], %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
// CHECK: }
// sync operands: column=2, row=0, direction=0, channel=3, column_num=1, row_num=1
// CHECK-DAG: %[[COL2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CHAN2:.*]] = arith.constant 3 : i32
// CHECK: aiex.npu.sync(%[[COL2]], %{{.*}}, %{{.*}}, %[[CHAN2]], %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_2_0 = aie.tile(2, 0)
    aie.runtime_sequence @await_loop_iter_arg(%arg0: memref<1024xi32>, %n: index) {
      // An unrelated task, on a different tile/direction/channel, with no SSA
      // connection to the loop's task chain below -- proves the resolution
      // below can't be coincidentally satisfied by wandering onto this one.
      %decoy = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 2 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%decoy)
      aiex.dma_await_task(%decoy)

      %c1 = arith.constant 1 : index
      %init = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
        %t = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
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

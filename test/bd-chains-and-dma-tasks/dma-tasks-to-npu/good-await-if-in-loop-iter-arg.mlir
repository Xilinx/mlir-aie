//===- good-await-if-in-loop-iter-arg.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// Mixed scf.for/scf.if nesting: each iteration picks the NEXT task via an
// scf.if (either branch configures a fresh task), and the loop awaits the
// PREVIOUS iteration's task -- carried in its own iter_arg -- before that
// pick runs. Resolving that await has to cross both kinds of region-branch
// boundary in one walk: the iter_arg's back-edge predecessor is the loop
// body's yielded value, which is itself an scf.if RESULT, so the walk must
// then follow the if's own successor-input mapping into both of its
// branches to reach a configure. good-await-loop-iter-arg.mlir and
// good-await-nested-loop-iter-arg.mlir only exercise homogeneous scf.for
// nesting; this is the first test to cross a for/if boundary.
//
// All three configures (%init, %ta, %tb) share tile (2, 0) / S2MM / channel 3
// -- required for correctness, since one static await site covers every
// iteration regardless of which branch ran -- distinct from the unrelated
// decoy on tile (0, 0) / MM2S / channel 0.

// CHECK-LABEL: @await_if_in_loop_iter_arg
// decoy: sync operands: column=0, row=0, direction=1, channel=0, column_num=1, row_num=1
// CHECK: aiex.npu.sync
// CHECK: scf.for
// sync operands: column=2, row=0, direction=0, channel=3, column_num=1, row_num=1
// CHECK-DAG: %[[COL:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CHAN:.*]] = arith.constant 3 : i32
// CHECK: aiex.npu.sync(%[[COL]], %{{.*}}, %{{.*}}, %[[CHAN]], %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
// CHECK: scf.if
// CHECK: }

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_2_0 = aie.tile(2, 0)
    aie.runtime_sequence @await_if_in_loop_iter_arg(%arg0: memref<1024xi32>, %n: index, %cond: i1) {
      // An unrelated task, on a different tile/direction/channel, with no SSA
      // connection to the loop/if task chain below.
      %decoy = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 3 : i32}
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
        aiex.dma_await_task(%prev)
        %t = scf.if %cond -> (index) {
          %ta = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
            aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256) {bd_id = 1 : i32}
            aie.end
          } {issue_token = true}
          aiex.dma_start_task(%ta)
          scf.yield %ta : index
        } else {
          %tb = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
            aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256) {bd_id = 2 : i32}
            aie.end
          } {issue_token = true}
          aiex.dma_start_task(%tb)
          scf.yield %tb : index
        }
        scf.yield %t : index
      }
      aiex.dma_await_task(%last)
    }
  }
}

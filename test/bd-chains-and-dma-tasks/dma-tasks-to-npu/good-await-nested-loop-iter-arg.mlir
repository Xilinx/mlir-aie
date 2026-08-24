//===- good-await-nested-loop-iter-arg.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu %s | FileCheck %s

// The iter_arg-await pattern from good-await-loop-iter-arg.mlir, nested two
// scf.for loops deep: the outer loop's iter_arg (itself carried in from the
// pre-loop init) feeds the inner loop's init, and the inner loop awaits its
// OWN iter_arg from inside its body. Resolving the inner await has to walk
// back across two levels of region-branch predecessors (inner iter_arg ->
// outer iter_arg -> pre-loop configure) to reach the originating configure.
//
// Going through the dynamic BD pool pass also exercises a second, adjacent
// fix: the pool pairs each task index with a runtime bd_id in the same
// iter_arg slot, and once the awaits lower to npu.sync the task-index half of
// every slot is dead end to end -- but only OBSERVABLY dead once you look
// through both loops together, since each loop in isolation still sees a
// genuine SSA use feeding the other. Local scf canonicalization cannot see
// that; only a cross-region liveness pass (--remove-dead-values, run after
// the local canonicalization) prunes it, which is why both loops below are
// expected to keep only their bd_id iter_arg and drop the task-index one
// entirely.
//
// The loop chain (%init, %t) uses tile (2, 0) / S2MM / channel 3 -- distinct
// from the unrelated decoy task on tile (0, 0) / MM2S / channel 0 -- so a
// resolution bug that stopped one hop too early (landing on the decoy or on
// nothing distinguishable) produces different, FileCheck-visible values.

// CHECK-LABEL: @nested
// decoy: sync operands: column=0, row=0, direction=1, channel=0, column_num=1, row_num=1
// CHECK: aiex.npu.sync
// CHECK: %[[INIT_ID:.*]] = aiex.dma_bd_pool_pop
// CHECK: scf.for {{.*}} iter_args(%[[OUTER_ID:.*]] = %[[INIT_ID]]) -> (i32) {
// CHECK:   scf.for {{.*}} iter_args(%[[INNER_ID:.*]] = %[[OUTER_ID]]) -> (i32) {
// sync operands: column=2, row=0, direction=0, channel=3, column_num=1, row_num=1
// CHECK-DAG:     %[[COL:.*]] = arith.constant 2 : i32
// CHECK-DAG:     %[[CHAN:.*]] = arith.constant 3 : i32
// CHECK:     aiex.npu.sync(%[[COL]], %{{.*}}, %{{.*}}, %[[CHAN]], %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
// CHECK:     aiex.dma_bd_pool_push({{.*}}) bd_id %[[INNER_ID]] : i32
// CHECK:   }
// CHECK: }
// sync operands: column=2, row=0, direction=0, channel=3, column_num=1, row_num=1
// CHECK-DAG: %[[COL2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CHAN2:.*]] = arith.constant 3 : i32
// CHECK: aiex.npu.sync(%[[COL2]], %{{.*}}, %{{.*}}, %[[CHAN2]], %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_2_0 = aie.tile(2, 0)
    aie.runtime_sequence @nested(%arg0: memref<1024xi32>, %n: index, %m: index) {
      // An unrelated task, on a different tile/direction/channel, with no SSA
      // connection to the nested loops' task chain below -- proves the
      // resolution below can't be coincidentally satisfied by wandering onto
      // this one, or by stopping one hop of nesting too early.
      %decoy = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%decoy)
      aiex.dma_await_task(%decoy)

      %c1 = arith.constant 1 : index
      %init = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%init)
      %outer_last = scf.for %i = %c1 to %n step %c1 iter_args(%outer_prev = %init) -> (index) {
        %inner_last = scf.for %j = %c1 to %m step %c1 iter_args(%inner_prev = %outer_prev) -> (index) {
          %t = aiex.dma_configure_task(%tile_2_0, S2MM, 3) {
            aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
            aie.end
          } {issue_token = true}
          aiex.dma_start_task(%t)
          aiex.dma_await_task(%inner_prev)
          aiex.dma_free_task(%inner_prev)
          scf.yield %t : index
        }
        scf.yield %inner_last : index
      }
      aiex.dma_await_task(%outer_last)
      aiex.dma_free_task(%outer_last)
    }
  }
}

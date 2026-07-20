//===- good-pingpong.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A rolled ping-pong over a RUNTIME-bound loop -- the exact form the static
// allocator rejects (see assign-runtime-sequence-bd-ids/bad-runtime-bound-
// pingpong.mlir). The dynamic pool pass keeps the loop rolled and turns the
// implicit allocation into runtime pool pop/push:
//   - each dma_configure_task gets a dma_bd_pool_pop feeding its bd_id_val;
//   - the popped id is carried through scf.for alongside the task (a second
//     i32 iter_arg), so the free of the PREVIOUS iteration's task pushes the
//     right id;
//   - the free after the loop pushes the last id; the redundant free-after-
//     await is dropped;
//   - the await, which named the loop-result task (no defining configure to
//     resolve), is redirected to the loop-invariant init configure %[[INIT_T]]
//     -- same tile/dir/channel every iteration.

// CHECK-LABEL: @runtime_bound_pingpong
// CHECK: %[[INIT:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK: %[[INIT_T:.*]] = aiex.dma_configure_task(%{{.*}}, MM2S, 0) bd_id %[[INIT]] : i32
// CHECK: %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args(%[[PREVT:.*]] = %[[INIT_T]], %[[PREVID:.*]] = %[[INIT]]) -> (index, i32)
// CHECK:   %[[T:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   aiex.dma_configure_task(%{{.*}}, MM2S, 0) bd_id %[[T]] : i32
// CHECK:   aiex.dma_bd_pool_push(0, 0) bd_id %[[PREVID]] : i32
// CHECK:   scf.yield %{{.*}}, %[[T]] : index, i32
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[LOOP]]#1 : i32
// CHECK: aiex.dma_await_task(%[[INIT_T]])

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @runtime_bound_pingpong(%arg0: memref<1024xi32>,
                                               %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}

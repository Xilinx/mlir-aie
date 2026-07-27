//===- good-for-in-if.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A runtime scf.for nested inside a branch of a runtime scf.if. The loop carries
// its ping-pong id through the for's iter_args (innermost-first: the for is
// handled before the enclosing if); the whole loop is self-contained in the
// branch, so the if gains no carried result.

// The for carries its ping-pong id through a second (i32) iter_arg; the whole
// loop is self-contained in the branch, so the if itself yields no results
// (it stays `scf.if %cond {`, never `scf.if %cond -> (...)`).
// CHECK-LABEL: @for_in_if
// CHECK: scf.if %{{.*}} {
// CHECK:   %[[INIT:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   aiex.dma_configure_task(%{{.*}}, MM2S, 0) bd_id %[[INIT]] : i32
// CHECK:   %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args({{.*}}, %[[PID:.*]] = %[[INIT]]) -> (index, i32)
// CHECK:     aiex.dma_bd_pool_push(0, 0) bd_id %[[PID]] : i32
// CHECK:   aiex.dma_bd_pool_push(0, 0) bd_id %[[LOOP]]#1 : i32
// CHECK-NOT: scf.if {{.*}} ->

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_in_if(%arg0: memref<1024xi32>, %n: index,
                                  %cond: i1) {
    %c1 = arith.constant 1 : index
    scf.if %cond {
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%t)
        aiex.dma_free_task(%prev)
        scf.yield %t : index
      }
      aiex.dma_await_task(%last)
      aiex.dma_free_task(%last)
    }
  }
}

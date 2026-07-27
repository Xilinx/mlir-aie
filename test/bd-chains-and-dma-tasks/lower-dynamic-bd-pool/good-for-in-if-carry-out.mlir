//===- good-for-in-if-carry-out.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A runtime scf.for nested in the then-branch of a runtime scf.if, whose task is
// carried OUT of both the loop and the if, then freed after the if. This needs
// the carry to compose across a nested boundary: the inner for grows an id
// iter_arg, and the enclosing if grows an id result that yields the loop's id on
// the then side and the branch-local pop on the else side. One pop per branch
// (one at runtime), one push after the if.

// CHECK-LABEL: @for_in_if_carry_out
// CHECK: %[[IF:.*]]:2 = scf.if %{{.*}} -> (index, i32) {
// CHECK:   %[[TID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   %[[LOOP:.*]]:2 = scf.for {{.*}} iter_args({{.*}}, %{{.*}} = %[[TID]]) -> (index, i32)
// CHECK:   scf.yield %[[LOOP]]#0, %[[LOOP]]#1 : index, i32
// CHECK: } else {
// CHECK:   %[[EID:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK:   scf.yield %{{.*}}, %[[EID]] : index, i32
// CHECK: }
// CHECK: aiex.dma_bd_pool_push(0, 0) bd_id %[[IF]]#1 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @for_in_if_carry_out(%arg0: memref<1024xi32>, %n: index,
                                            %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r = scf.if %cond -> (index) {
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      %l = scf.for %i = %c0 to %n step %c1 iter_args(%tk = %init) -> (index) {
        aiex.dma_start_task(%tk)
        aiex.dma_await_task(%tk)
        scf.yield %tk : index
      }
      scf.yield %l : index
    } else {
      %init2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%init2)
      aiex.dma_await_task(%init2)
      scf.yield %init2 : index
    }
    aiex.dma_free_task(%r)
  }
}

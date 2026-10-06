//===- good-multi-bd-chain.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A 2-BD chain under a runtime-bound loop pops one id per BD, in body order,
// from the channel's partition. Both ids ride the loop as parallel i32
// iter_args next to the task, and the free pushes both.

// CHECK-LABEL: @chain_pingpong
// CHECK: %[[I0:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK: %[[I1:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK: %[[INIT_T:.*]] = aiex.dma_configure_task(%{{.*}}, MM2S, 0) {
// CHECK:   aie.dma_bd({{.*}} offset = 0 len = 128) bd_id_val %[[I0]] : i32
// CHECK:   aie.next_bd ^bb1
// CHECK:   aie.dma_bd({{.*}} offset = 128 len = 128) bd_id_val %[[I1]] : i32
// CHECK: %[[LOOP:.*]]:3 = scf.for {{.*}} iter_args(%{{.*}} = %[[INIT_T]], %[[P0:.*]] = %[[I0]], %[[P1:.*]] = %[[I1]]) -> (index, i32, i32)
// CHECK:   %[[T0:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   %[[T1:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   aie.dma_bd({{.*}} offset = 0 len = 128) bd_id_val %[[T0]] : i32
// CHECK:   aie.dma_bd({{.*}} offset = 128 len = 128) bd_id_val %[[T1]] : i32
// CHECK:   aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[P0]] : i32
// CHECK:   aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[P1]] : i32
// CHECK:   scf.yield %{{.*}}, %[[T0]], %[[T1]] : index, i32, i32
// CHECK: aiex.dma_await_task(%[[LOOP]]#0)
// CHECK: aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[LOOP]]#1 : i32
// CHECK: aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[LOOP]]#2 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @chain_pingpong(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 128)
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 128 len = 128)
      aie.end
    }
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 128)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 128 len = 128)
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

// -----

// A chain carried out of an scf.if: each branch yields its own task and its
// own two ids, laid out after the original results, and the post-if free
// pushes both carried ids. A single-BD task carried through the same if keeps
// its one id, after the chain's two.

// CHECK-LABEL: @chain_if_carry
// CHECK: %[[R:.*]]:5 = scf.if %{{.*}} -> (index, index, i32, i32, i32) {
// CHECK:   %[[A0:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   %[[A1:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   %[[AS:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   scf.yield %{{.*}}, %{{.*}}, %[[A0]], %[[A1]], %[[AS]] : index, index, i32, i32, i32
// CHECK: } else {
// CHECK:   %[[B0:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   %[[B1:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   %[[BS:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK:   scf.yield %{{.*}}, %{{.*}}, %[[B0]], %[[B1]], %[[BS]] : index, index, i32, i32, i32
// CHECK: }
// CHECK: aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[R]]#2 : i32
// CHECK: aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[R]]#3 : i32
// CHECK: aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %[[R]]#4 : i32

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @chain_if_carry(%arg0: memref<1024xi32>, %cond: i1) {
    %r:2 = scf.if %cond -> (index, index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 128)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 128 len = 128)
        aie.end
      } {issue_token = true}
      %s = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 128)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_start_task(%s)
      aiex.dma_await_task(%t)
      aiex.dma_await_task(%s)
      scf.yield %t, %s : index, index
    } else {
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 128)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 640 len = 128)
        aie.end
      } {issue_token = true}
      %s2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 768 len = 128)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_start_task(%s2)
      aiex.dma_await_task(%t2)
      aiex.dma_await_task(%s2)
      scf.yield %t2, %s2 : index, index
    }
    aiex.dma_free_task(%r#0)
    aiex.dma_free_task(%r#1)
  }
}

// -----

// A mem tile chain draws every id from its channel's parity partition, and
// each BD gets its own pop.

// CHECK-LABEL: @memtile_chain
// CHECK: %[[M0:.*]] = aiex.dma_bd_pool_pop(0, 1) partition [24, 48) : i32
// CHECK: %[[M1:.*]] = aiex.dma_bd_pool_pop(0, 1) partition [24, 48) : i32
// CHECK: %[[M2:.*]] = aiex.dma_bd_pool_pop(0, 1) partition [24, 48) : i32
// CHECK: aie.dma_bd({{.*}} bd_id_val %[[M0]] : i32
// CHECK: aie.dma_bd({{.*}} bd_id_val %[[M1]] : i32
// CHECK: aie.dma_bd({{.*}} bd_id_val %[[M2]] : i32
// CHECK: aiex.dma_bd_pool_push(0, 1) partition [24, 48) bd_id %[[M0]] : i32
// CHECK: aiex.dma_bd_pool_push(0, 1) partition [24, 48) bd_id %[[M1]] : i32
// CHECK: aiex.dma_bd_pool_push(0, 1) partition [24, 48) bd_id %[[M2]] : i32

aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) : memref<1024xi32>
  aie.runtime_sequence @memtile_chain(%n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%buf : memref<1024xi32> offset = 256 len = 256)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%buf : memref<1024xi32> offset = 512 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}

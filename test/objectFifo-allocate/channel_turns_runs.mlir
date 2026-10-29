// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=CHAIN

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The core takes one of a before its loop and one of b after it, a and b in
// turn inside. The channel repeats the loop's turns alone: the pools' depths
// absorb the first and last steps.

// CHECK-LABEL: @steady_run
// CHECK-DAG: aie.objectfifo.dma_endpoint @a_cons_dma(%{{.*}}) fills @a_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @b_cons_dma(%{{.*}}) fills @b_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}
module @steady_run {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c(%s1, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c7 = arith.constant 7 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c64 step %c1 {
        %x = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @c(Consume, 1)
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 1)
        scf.for %j = %c0 to %c7 step %c1 {
          %a1 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
          %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
          aie.objectfifo.release @a(Consume, 1)
          aie.objectfifo.release @b(Consume, 1)
        }
        %b1 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @b(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A mem tile holds 40 of a, more than its descriptors can take one an object,
// so one descriptor a side walks the pool whole. The core takes two of a for
// each of b, so two descriptors walk a's pool in turn, each every other
// object, and b's pool, walked whole beside them, lines up with them.

// CHECK-LABEL: @two_walkers
// CHECK-DAG: aie.objectfifo.dma_endpoint @a_prod_dma(%{{.*}}) drains @a_in_cons_pool {{{.*}}turns = [@a_prod_dma, @a_prod_dma, @b_prod_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @b_prod_dma(%{{.*}}) drains @b_in_cons_pool {{{.*}}turns = [@a_prod_dma, @a_prod_dma, @b_prod_dma]}
// CHECK-DAG: aie.objectfifo.pool @b_in_cons_pool(%{{.*}}) {buffers = [@b_in_cons_buff_0], contiguous, depth = 3

// CHAIN-LABEL: @two_walkers
// CHAIN: aie.memtile_dma(%mem_tile_0_1)
// CHAIN: aie.dma_start(MM2S
// CHAIN: aie.dma_bd(%a_in_cons_buff_0 : memref<2560xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 20, stride = 128, current = 0>}
// CHAIN: aie.dma_bd(%a_in_cons_buff_0 : memref<2560xi32> offset = 64 len = 64) {iteration = #aie.bd_iteration<size = 20, stride = 128, current = 0>}
// CHAIN: aie.dma_bd(%b_in_cons_buff_0 : memref<192xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 3, stride = 64, current = 0>}
// CHAIN: aie.next_bd ^bb5
module @two_walkers {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @a_in(%s, {%m}, [2 : i32, 40 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, [2 : i32, 3 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c(%s1, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c40 = arith.constant 40 : index
      scf.for %i = %c0 to %c40 step %c1 {
        %x = aie.objectfifo.acquire @c(Consume, 1) : memref<64xi32>
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @a(Consume, 1)
        %a1 = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @a(Consume, 1)
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// The core takes a and b in turn from pools of 7 and 6 objects. A chain over
// them one object a descriptor would unroll to lcm(7, 6) passes, 84
// descriptors, more than a mem tile channel has, so both pools are walked
// whole, one descriptor a turn.

// CHECK-LABEL: @unrolled_chain
// CHECK-DAG: aie.objectfifo.dma_endpoint @a_prod_dma(%{{.*}}) drains @a_in_cons_pool {{{.*}}turns = [@a_prod_dma, @b_prod_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @b_prod_dma(%{{.*}}) drains @b_in_cons_pool {{{.*}}turns = [@a_prod_dma, @b_prod_dma]}
// CHECK-DAG: aie.objectfifo.pool @a_in_cons_pool(%{{.*}}) {buffers = [@a_in_cons_buff_0], contiguous, depth = 7
// CHECK-DAG: aie.objectfifo.pool @b_in_cons_pool(%{{.*}}) {buffers = [@b_in_cons_buff_0], contiguous, depth = 6

// CHAIN-LABEL: @unrolled_chain
// CHAIN: aie.memtile_dma(%mem_tile_0_1)
// CHAIN: aie.dma_start(MM2S
// CHAIN: aie.dma_bd(%a_in_cons_buff_0 : memref<448xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 7, stride = 64, current = 0>}
// CHAIN: aie.dma_bd(%b_in_cons_buff_0 : memref<384xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 6, stride = 64, current = 0>}
// CHAIN: aie.next_bd ^bb
module @unrolled_chain {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @a_in(%s, {%m}, [2 : i32, 7 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, [2 : i32, 6 : i32]) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c(%s1, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c42 = arith.constant 42 : index
      scf.for %i = %c0 to %c42 step %c1 {
        %x = aie.objectfifo.acquire @c(Consume, 1) : memref<64xi32>
        %a0 = aie.objectfifo.acquire @a(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @a(Consume, 1)
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<64xi32>
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}

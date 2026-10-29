// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=CHAIN

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A core reads three fifos from one mem tile over two input channels, so two
// of them take turns on one, in the order the core acquires them. One circuit
// carries both, and the mem tile ends feeding it take the same turns.

// CHECK-LABEL: @three_fifos_two_channels
// CHECK: aie.objectfifo.dma_endpoint @a_cons_dma(%{{.*}}) fills @a_cons_pool {channelIndex = {{[0-9]+}} : i32, fifoName = "a"}
// CHECK: aie.objectfifo.dma_endpoint @b_prod_dma(%{{.*}}) drains @bc_in_cons_pool {channelIndex = [[MM2S:[0-9]+]] : i32, {{.*}}turns = [@b_prod_dma, @c_prod_dma]}
// CHECK: aie.objectfifo.dma_endpoint @b_cons_dma(%{{.*}}) fills @b_cons_pool {channelIndex = [[S2MM:[0-9]+]] : i32, {{.*}}turns = [@b_cons_dma, @c_cons_dma]}
// CHECK: aie.flow(%mem_tile_0_1, DMA : [[MM2S]], %tile_0_2, DMA : [[S2MM]])
// CHECK: aie.objectfifo.dma_endpoint @c_prod_dma(%{{.*}}) drains @bc_in_cons_pool {channelIndex = [[MM2S]] : i32, {{.*}}turns = [@b_prod_dma, @c_prod_dma]}
// CHECK: aie.objectfifo.dma_endpoint @c_cons_dma(%{{.*}}) fills @c_cons_pool {channelIndex = [[S2MM]] : i32, {{.*}}turns = [@b_cons_dma, @c_cons_dma]}
// CHECK-NOT: aie.flow(%mem_tile_0_1

// The core's chain moves b and c in turn, each through its own pool.
// CHAIN-LABEL: @three_fifos_two_channels
// CHAIN: aie.mem(%tile_0_2)
// CHAIN: aie.dma_start(S2MM
// CHAIN: aie.dma_bd(%b_cons_buff_0
// CHAIN: aie.dma_bd(%c_cons_buff_0
// CHAIN: aie.dma_bd(%b_cons_buff_1
// CHAIN: aie.dma_bd(%c_cons_buff_1
// CHAIN: aie.next_bd ^bb1
// CHAIN: aie.dma_start(S2MM
// CHAIN-NOT: %c_cons_buff
// CHAIN: aie.dma_bd(%a_cons_buff_0
module @three_fifos_two_channels {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @bc_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@bc_in] -> [@b, @c] ([] [0, 16])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %a0, %a1 = aie.objectfifo.acquire @a(Consume, 2) : memref<16xi32>, memref<16xi32>
        %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
        aie.objectfifo.release @a(Consume, 2)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// A window over a (three held, one released a step) next to b: a channel
// moving a and b in turn hands the core a's first two objects, then b's
// first, and b's second needs a slot b can only free after the core has a's
// third. With b two deep the turns keep up, so they are taken.

// CHECK-LABEL: @window_beside_two_deep
// CHECK-DAG: aie.objectfifo.dma_endpoint @a_cons_dma(%{{.*}}) fills @a_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @b_cons_dma(%{{.*}}) fills @b_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @c_cons_dma(%{{.*}}) fills @c_cons_pool {channelIndex = {{[0-9]+}} : i32, fifoName = "c"}
module @window_beside_two_deep {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @b_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c_in(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@c_in] -> [@c] ([] [])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c64 step %c1 {
        %a0, %a1, %a2 = aie.objectfifo.acquire @a(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
        %b0 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %first = arith.cmpi eq, %i, %c0 : index
        scf.if %first {
          %c0_ = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
          aie.objectfifo.release @c(Consume, 1)
        }
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @a(Consume, 1)
      }
      aie.end
    }
  }
}

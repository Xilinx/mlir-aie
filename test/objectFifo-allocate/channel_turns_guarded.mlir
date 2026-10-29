// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=CHAIN

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A core runs its steps as one loop and acquires a in steps 0-3 and b in steps
// 2-5, under branches on the step counter. The order is still fixed: the loop
// around the steps is the period, the steps unroll and the branches fold, so a
// and b take turns on one input channel.

// CHECK-LABEL: @rolled_windows
// CHECK-DAG: aie.objectfifo.dma_endpoint @a_cons_dma(%{{.*}}) fills @a_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @b_cons_dma(%{{.*}}) fills @b_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @c_cons_dma(%{{.*}}) fills @c_cons_pool {channelIndex = {{[0-9]+}} : i32, fifoName = "c"}

// CHAIN-LABEL: @rolled_windows
// CHAIN: aie.mem(%tile_0_2)
// CHAIN: aie.dma_start(S2MM
// CHAIN: aie.dma_bd(%a_cons_buff_0
// CHAIN: aie.dma_bd(%b_cons_buff_0
// CHAIN: aie.dma_bd(%a_cons_buff_1
// CHAIN: aie.dma_bd(%b_cons_buff_1
// CHAIN: aie.next_bd ^bb1
module @rolled_windows {
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
    aie.objectfifo @c_in(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@c_in] -> [@c] ([] [])
    aie.core(%t) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c6 = arith.constant 6 : index
        scf.for %step = %c0 to %c6 step %c1 {
          %ra = arith.cmpi ult, %step, %c4 : index
          scf.if %ra {
            %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
            aie.objectfifo.release @a(Consume, 1)
          }
          %from = arith.cmpi uge, %step, %c2 : index
          %to = arith.cmpi ult, %step, %c6 : index
          %rb = arith.andi %from, %to : i1
          scf.if %rb {
            %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
            aie.objectfifo.release @b(Consume, 1)
          }
          %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
          aie.objectfifo.release @c(Consume, 1)
        }
        scf.yield
      }
      aie.end
    }
  }
}

// -----

// Steps alternate between a and b, the branch on the step counter's parity
// taking its else side on odd steps.

// CHECK-LABEL: @alternate_steps
// CHECK-DAG: aie.objectfifo.dma_endpoint @a_cons_dma(%{{.*}}) fills @a_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @b_cons_dma(%{{.*}}) fills @b_cons_pool {{{.*}}turns = [@a_cons_dma, @b_cons_dma]}

// CHAIN-LABEL: @alternate_steps
// CHAIN: aie.mem(%tile_0_2)
// CHAIN: aie.dma_start(S2MM
// CHAIN: aie.dma_bd(%a_cons_buff_0
// CHAIN: aie.dma_bd(%b_cons_buff_0
// CHAIN: aie.dma_bd(%a_cons_buff_1
// CHAIN: aie.dma_bd(%b_cons_buff_1
// CHAIN: aie.next_bd ^bb1
module @alternate_steps {
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
    aie.objectfifo @c_in(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@c_in] -> [@c] ([] [])
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      scf.for %image = %c0 to %c64 step %c1 {
        scf.for %step = %c0 to %c8 step %c1 {
          %parity = arith.remsi %step, %c2 : index
          %even = arith.cmpi eq, %parity, %c0 : index
          scf.if %even {
            %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
            aie.objectfifo.release @a(Consume, 1)
          } else {
            %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
            aie.objectfifo.release @b(Consume, 1)
          }
          %c = aie.objectfifo.acquire @c(Consume, 1) : memref<16xi32>
          aie.objectfifo.release @c(Consume, 1)
        }
      }
      aie.end
    }
  }
}

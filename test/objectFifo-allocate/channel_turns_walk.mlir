// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// b and c reach the core in turns on one channel, and their mem tile ends
// take the same turns: one channel, counted once wherever channels are
// planned. @deep's 30 objects then leave the tile short of BD ids, so @deep
// is walked whole, and the turns stand.
// CHECK-LABEL: @turns_and_walk
// CHECK-DAG: aie.objectfifo.dma_endpoint @b__prod__dma(%{{.*}}) drains @bc_in__cons__pool {channelIndex = [[MM2S:[0-9]+]] : i32, {{.*}}turns = [@b__prod__dma, @c__prod__dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @c__prod__dma(%{{.*}}) drains @bc_in__cons__pool {channelIndex = [[MM2S]] : i32, {{.*}}turns = [@b__prod__dma, @c__prod__dma]}
// CHECK-DAG: aie.objectfifo.pool @deep__cons__pool(%{{.*}}) {{{.*}}contiguous, depth = 30 : i32
module @turns_and_walk {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 2)
    %s1 = aie.tile(1, 0)
    aie.objectfifo @a_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@a_in] -> [@a] ([] [])
    aie.objectfifo @bc_in(%s, {%m}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@bc_in] -> [@b, @c] ([] [0, 16])
    aie.objectfifo @deep(%s1, {%m}, 30 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @deep_out(%m, {%s1}, 30 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@deep] -> [@deep_out] ([] [])
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

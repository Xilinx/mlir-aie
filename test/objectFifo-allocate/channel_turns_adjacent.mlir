// RUN: aie-opt --aie-objectfifo-split --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie.device(aie-objectFifo-stateful-transform,aie-assign-buffer-addresses))' %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// b and c reach the core in turns, so their mem tile ends take turns on one
// MM2S channel, b's. @full leaves room at home for b's buffers but not c's,
// which spill to the neighbors. The shared channel then runs BDs on c's
// buffers too, so it must be one of the channels that reach adjacent memory,
// although b's own buffers are local.
// CHECK-LABEL: @turn_reaches_spilled_buffer
// CHECK-DAG: %c_in__cons__buf0 = aie.buffer(%mem_tile_0_1)
// CHECK-DAG: %c_in__cons__buf1 = aie.buffer(%mem_tile_2_1)
// CHECK-DAG: %b_in__cons__buf0 = aie.buffer(%mem_tile_1_1)
// CHECK-DAG: aie.objectfifo.dma_endpoint @b__prod__dma(%mem_tile_1_1) drains @b_in__cons__pool {channelIndex = [[MM2S:[0-3]]] : i32, {{.*}}turns = [@b__prod__dma, @c__prod__dma]}
// CHECK-DAG: aie.objectfifo.dma_endpoint @c__prod__dma(%mem_tile_1_1) drains @c_in__cons__pool {channelIndex = [[MM2S]] : i32, {{.*}}turns = [@b__prod__dma, @c__prod__dma]}
module @turn_reaches_spilled_buffer {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %m = aie.tile(1, 1)
    %t = aie.tile(1, 2)
    %u = aie.tile(1, 3)
    %full = aie.buffer(%m) {sym_name = "full"} : memref<523264xi8>
    aie.objectfifo @a(%s2, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b_in(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@b_in] -> [@b] ([] [])
    aie.objectfifo @c_in(%s1, {%m}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @c(%m, {%t}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [@c_in] -> [@c] ([] [])
    aie.objectfifo @w(%m, {%u}, 1 : i32) : !aie.objectfifo<memref<16xi32>> = [dense<1> : memref<16xi32>]
    aie.objectfifo @d_in(%s0, {%m}, 2 : i32) : !aie.objectfifo<memref<12288xi32>>
    aie.objectfifo @d(%m, {%s0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @e(%m, {%s0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo @f(%m, {%s2}, 2 : i32) : !aie.objectfifo<memref<4096xi32>>
    aie.objectfifo.link [@d_in] -> [@d, @e, @f] ([] [0, 4096, 8192])
    aie.core(%u) {
      %w = aie.objectfifo.acquire @w(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @w(Consume, 1)
      aie.end
    }
    aie.core(%t) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
        %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
        %c = aie.objectfifo.acquire @c(Consume, 1) : memref<256xi32>
        aie.objectfifo.release @a(Consume, 1)
        aie.objectfifo.release @b(Consume, 1)
        aie.objectfifo.release @c(Consume, 1)
      }
      aie.end
    }
  }
}

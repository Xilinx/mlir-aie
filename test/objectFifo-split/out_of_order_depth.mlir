//===- out_of_order_depth.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A core's DMA pool keeps what its largest acquire needs plus one, but an
// out-of-order fifo keeps the depth it asks for at both ends: each sending
// buffer names a receiving one, and the depth is what keeps a channel it
// shares from stalling on a full pool while its core waits on another fifo's
// packet queued behind it.

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// CHECK-LABEL: module @kept
// CHECK-DAG: aie.objectfifo.pool @a__pool(%{{.*}}) {depth = 4 : i32
// CHECK-DAG: aie.objectfifo.pool @a__cons__pool(%{{.*}}) {depth = 4 : i32
// A circuit asking for as many keeps only what its cores' acquires need.
// CHECK-DAG: aie.objectfifo.pool @b__pool(%{{.*}}) {depth = 2 : i32
// CHECK-DAG: aie.objectfifo.pool @b__cons__pool(%{{.*}}) {depth = 2 : i32
module @kept {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_5 = aie.tile(2, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_2_5}, 4 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_1_3, {%tile_2_5}, 4 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %1 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %1 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
    %core_2_5 = aie.core(%tile_2_5) {
      %1 = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      %2 = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
  }
}

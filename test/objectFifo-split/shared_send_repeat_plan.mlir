// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Two fifos pin one core MM2S channel and repeat their objects. The split
// gives both push_queue, which repeats in each pushed task, and the plan the
// placers take from plannedDmaEnd agrees: depth BDs each, not depth * repeat.
// CHECK-LABEL: module @implicit_push_queue_repeat
// CHECK: aie.objectfifo.dma_endpoint @a__prod__dma({{.*}}) drains @a__pool {{{.*}}release_strategy = #aie.release_strategy<push_queue>
// CHECK: aie.objectfifo.dma_endpoint @b__prod__dma({{.*}}) drains @b__pool {{{.*}}release_strategy = #aie.release_strategy<push_queue>
module @implicit_push_queue_repeat {
  aie.device(npu2) {
    %src = aie.tile(0, 2)
    %d0 = aie.tile(0, 4)
    %d1 = aie.tile(1, 4)
    aie.objectfifo @a(%src, {%d0}, [2, 2]) {repeat_count = 3 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%src, {%d1}, [2, 2]) {repeat_count = 3 : i32, transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%src) {
      %a0 = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b0 = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
  }
}

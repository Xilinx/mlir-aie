// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// Three packet FIFOs pinned to one core MM2S channel use one output channel,
// as both placers count them, so they fit without an over-capacity warning.
// CHECK-LABEL: module @shared_core_channel
// CHECK-DAG: aie.objectfifo.dma_endpoint @a__prod__dma{{.*}}channelIndex = 0{{.*}}release_strategy = #aie.release_strategy<push_queue>
// CHECK-DAG: aie.objectfifo.dma_endpoint @b__prod__dma{{.*}}channelIndex = 0{{.*}}release_strategy = #aie.release_strategy<push_queue>
// CHECK-DAG: aie.objectfifo.dma_endpoint @c__prod__dma{{.*}}channelIndex = 0{{.*}}release_strategy = #aie.release_strategy<push_queue>

module @shared_core_channel {
  aie.device(npu2) {
    %core_tile = aie.logical_tile<CoreTile>(?, ?)
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    %s2 = aie.logical_tile<ShimNOCTile>(2, 0)
    aie.objectfifo @a(%core_tile, {%s0}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%core_tile, {%s1}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%core_tile, {%s2}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%core_tile) {
      %a = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      %c = aie.objectfifo.acquire @c(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @c(Produce, 1)
      aie.end
    }
  }
}

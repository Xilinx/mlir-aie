//===- test_sa_pinned_channels.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ends that pin one DMA channel of one tile and may share it take one channel,
// as the sequential placer and objectFIFO allocation count them, so three of
// them fit a core's two channels without an over-capacity warning.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=2' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=3' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// Three out-of-order fifos pinned to one S2MM channel of a receiving core.
// CHECK-LABEL: module @shared_core_receive
// CHECK-DAG: aie.objectfifo.dma_endpoint @fa__cons__dma{{.*}}channelIndex = 1
// CHECK-DAG: aie.objectfifo.dma_endpoint @fb__cons__dma{{.*}}channelIndex = 1
// CHECK-DAG: aie.objectfifo.dma_endpoint @fc__cons__dma{{.*}}channelIndex = 1
module @shared_core_receive {
  aie.device(npu2) {
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %r = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @fa(%a, {%r}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fb(%b, {%r}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fc(%c, {%r}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 5>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %1 = aie.objectfifo.acquire @fa(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %1 = aie.objectfifo.acquire @fb(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fb(Produce, 1)
      aie.end
    }
    aie.core(%c) {
      %1 = aie.objectfifo.acquire @fc(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fc(Produce, 1)
      aie.end
    }
    aie.core(%r) {
      %1 = aie.objectfifo.acquire @fa(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Consume, 1)
      %2 = aie.objectfifo.acquire @fb(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @fb(Consume, 1)
      %3 = aie.objectfifo.acquire @fc(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @fc(Consume, 1)
      aie.end
    }
  }
}

// -----

// The same three sending ends on a physical core: it takes the one channel
// too, with no over-capacity warning.
// CHECK-LABEL: module @shared_physical_core_send
// CHECK-DAG: aie.objectfifo.dma_endpoint @a__prod__dma{{.*}}channelIndex = 0{{.*}}release_strategy = #aie.release_strategy<push_queue>
// CHECK-DAG: aie.objectfifo.dma_endpoint @b__prod__dma{{.*}}channelIndex = 0{{.*}}release_strategy = #aie.release_strategy<push_queue>
// CHECK-DAG: aie.objectfifo.dma_endpoint @c__prod__dma{{.*}}channelIndex = 0{{.*}}release_strategy = #aie.release_strategy<push_queue>
module @shared_physical_core_send {
  aie.device(npu2) {
    %core_tile = aie.tile(0, 2)
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

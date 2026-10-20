//===- fan_in.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-lower-dmas %s | FileCheck %s --check-prefix=DMAS
// RUN: aie-opt --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-lower-dmas --aie-objectfifo-lower-cores --aie-objectfifo-erase-pools --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=ROUTED

// Two producers feed one consumer's pool. The route lowers to one packet flow
// with a source per producer under one header, every producer's descriptors
// stamp that header, and the consumer's chain is the ordinary ring: whichever
// packet arrives fills the next object. The pathfinder lays down both paths.

// CHECK-LABEL: @fan_in
// CHECK:       aie.objectfifo.dma_endpoint @a_dma({{.*}}) drains @a_pool {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_id = 0>}
// CHECK:       aie.objectfifo.dma_endpoint @b_dma({{.*}}) drains @b_pool {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_id = 0>}
// CHECK:       aie.objectfifo.dma_endpoint @c_dma({{.*}}) fills @c_pool {channelIndex = 0 : i32}
// CHECK:       aie.packet_flow(0) {
// CHECK-NEXT:    aie.packet_source<%[[A:.*]], DMA : 0>
// CHECK-NEXT:    aie.packet_source<%[[B:.*]], DMA : 0>
// CHECK-NEXT:    aie.packet_dest<%[[C:.*]], DMA : 0>
// CHECK-NOT:   aie.route

// DMAS:        aie.mem(%{{.*}}) {
// DMAS:          aie.dma_start(MM2S, 0
// DMAS:          aie.dma_bd_packet(0, 0)
// DMAS:          aie.dma_bd(%a_buff_0
// DMAS:        aie.mem(%{{.*}}) {
// DMAS:          aie.dma_start(MM2S, 0
// DMAS:          aie.dma_bd_packet(0, 0)
// DMAS:          aie.dma_bd(%b_buff_0
// DMAS:        aie.mem(%{{.*}}) {
// DMAS:          aie.dma_start(S2MM, 0
// DMAS-NOT:      aie.dma_bd_packet
// DMAS:          aie.dma_bd(%c_buff_0
// DMAS:          aie.dma_bd(%c_buff_1

// ROUTED:      aie.packet_rules
// ROUTED:      aie.packet_rules

module @fan_in {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile32 = aie.tile(3, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo.pool @a_pool(%tile12) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @a(%tile12) fills @a_pool
    aie.objectfifo.dma_endpoint @a_dma(%tile12) drains @a_pool

    aie.objectfifo.pool @b_pool(%tile32) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @b(%tile32) fills @b_pool
    aie.objectfifo.dma_endpoint @b_dma(%tile32) drains @b_pool

    aie.objectfifo.pool @c_pool(%tile23) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @c_dma(%tile23) fills @c_pool
    aie.objectfifo.core_endpoint @c(%tile23) drains @c_pool

    aie.route from [@a_dma, @b_dma] to [@c_dma] {packet = #aie.packet_info<>}

    %core12 = aie.core(%tile12) {
      %o = aie.objectfifo.acquire @a (1) : memref<16xi32>
      aie.objectfifo.release @a (1)
      aie.end
    }
    %core32 = aie.core(%tile32) {
      %o = aie.objectfifo.acquire @b (1) : memref<16xi32>
      aie.objectfifo.release @b (1)
      aie.end
    }
    %core24 = aie.core(%tile23) {
      %o = aie.objectfifo.acquire @c (1) : memref<16xi32>
      aie.objectfifo.release @c (1)
      aie.end
    }
  }
}

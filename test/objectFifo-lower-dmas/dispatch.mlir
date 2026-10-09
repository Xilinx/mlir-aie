// RUN: aie-opt --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s --check-prefix=ALLOC
// RUN: aie-opt --aie-objectfifo-verify --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-lower-dmas %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A mem tile pool of two objects dispatches to three cores in turn over one
// MM2S channel. Allocation records each turn's header, and the chain runs
// lcm(2, 3) = 6 objects so every object meets every turn: buffers 0 1 0 1 0 1
// carry the headers of a, b, c, a, b, c.

// ALLOC:       aie.objectfifo.dma_endpoint @y_out({{.*}}) drains @y {channelIndex = 0 : i32, dispatch = [@a_in, @b_in, @c_in], dispatchPackets = [#aie.packet_info<pkt_id = 0>, #aie.packet_info<pkt_id = 9>, #aie.packet_info<pkt_id = 1>]}
// ALLOC:       aie.packet_flow(0) {
// ALLOC-NEXT:    aie.packet_source<%[[MEM:.*]], DMA : 0>
// ALLOC-NEXT:    aie.packet_dest<%{{.*}}tile_0_2, DMA : 0>
// ALLOC:       aie.packet_flow(9) {
// ALLOC-NEXT:    aie.packet_source<%[[MEM]], DMA : 0>
// ALLOC-NEXT:    aie.packet_dest<%{{.*}}tile_0_3, DMA : 0>
// ALLOC:       aie.packet_flow(1) {
// ALLOC-NEXT:    aie.packet_source<%[[MEM]], DMA : 0>
// ALLOC-NEXT:    aie.packet_dest<%{{.*}}tile_0_4, DMA : 0>

// CHECK:       aie.memtile_dma
// CHECK:         aie.dma_start(MM2S, 0, ^[[BD0:.*]], ^
// CHECK:       ^[[BD0]]:
// CHECK:         aie.dma_bd_packet(0, 0)
// CHECK-NEXT:    aie.dma_bd(%y_buff_0
// CHECK:         aie.dma_bd_packet(0, 9)
// CHECK-NEXT:    aie.dma_bd(%y_buff_1
// CHECK:         aie.dma_bd_packet(0, 1)
// CHECK-NEXT:    aie.dma_bd(%y_buff_0
// CHECK:         aie.dma_bd_packet(0, 0)
// CHECK-NEXT:    aie.dma_bd(%y_buff_1
// CHECK:         aie.dma_bd_packet(0, 9)
// CHECK-NEXT:    aie.dma_bd(%y_buff_0
// CHECK:         aie.dma_bd_packet(0, 1)
// CHECK-NEXT:    aie.dma_bd(%y_buff_1
// CHECK:         aie.next_bd ^[[BD0]]

module @dispatch {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %c = aie.tile(0, 4)
    aie.route_endpoint @in(%shim) DMA {fifoName = "in"}
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @y_in(%mem) fills @y
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a_in, @b_in, @c_in]}
    aie.objectfifo.pool @pa(%a) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @a_in(%a) fills @pa
    aie.objectfifo.core_endpoint @a_core(%a) drains @pa
    aie.objectfifo.pool @pb(%b) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @b_in(%b) fills @pb
    aie.objectfifo.core_endpoint @b_core(%b) drains @pb
    aie.objectfifo.pool @pc(%c) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @c_in(%c) fills @pc
    aie.objectfifo.core_endpoint @c_core(%c) drains @pc
    aie.route from @in to [@y_in]
    aie.route from @y_out to [@a_in] {packet = #aie.packet_info<>}
    aie.route from @y_out to [@b_in] {packet = #aie.packet_info<pkt_id = 9>}
    aie.route from @y_out to [@c_in] {packet = #aie.packet_info<>}
  }
}

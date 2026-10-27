// RUN: aie-opt --split-input-file --aie-objectfifo-allocate %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One design written twice: the second declares its tiles, pools, endpoints
// and routes in another order and renames every symbol (a -> y, b -> x,
// c -> w, d -> v). Each object gets the same channel, packet ID and buffer
// order in both. @a and @b differ only past the MemTile, in the core they
// feed, and so do @c and @d in the core they drain.

// CHECK-LABEL: module @declared
// CHECK:       aie.buffer @a_l2_buff_0
// CHECK:       aie.buffer @b_l2_buff_0
// CHECK:       aie.buffer @c_l1_buff_0
// CHECK:       aie.buffer @a_l1_buff_0
// CHECK:       aie.buffer @d_l1_buff_0
// CHECK:       aie.buffer @b_l1_buff_0
// CHECK-DAG:   aie.route_endpoint @a_shim({{.*}}) DMA {channelIndex = 0 : i32, fifoName = "a"}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @a_in({{.*}}) fills @a_l2 {channelIndex = 0 : i32}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @a_out({{.*}}) drains @a_l2 {channelIndex = 0 : i32}
// CHECK-DAG:   aie.route_endpoint @b_shim({{.*}}) DMA {channelIndex = 1 : i32, fifoName = "b"}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @b_in({{.*}}) fills @b_l2 {channelIndex = 1 : i32}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @b_out({{.*}}) drains @b_l2 {channelIndex = 1 : i32}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @c_core({{.*}}) drains @c_l1 {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
// CHECK-DAG:   aie.route_endpoint @c_shim({{.*}}) DMA {channelIndex = 0 : i32, fifoName = "c"}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @d_core({{.*}}) drains @d_l1 {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK-DAG:   aie.route_endpoint @d_shim({{.*}}) DMA {channelIndex = 1 : i32, fifoName = "d"}

// CHECK-LABEL: module @permuted
// CHECK:       aie.buffer @v_l1_buff_0
// CHECK:       aie.buffer @x_l1_buff_0
// CHECK:       aie.buffer @w_l1_buff_0
// CHECK:       aie.buffer @y_l1_buff_0
// CHECK:       aie.buffer @y_l2_buff_0
// CHECK:       aie.buffer @x_l2_buff_0
// CHECK-DAG:   aie.route_endpoint @y_shim({{.*}}) DMA {channelIndex = 0 : i32, fifoName = "y"}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @y_in({{.*}}) fills @y_l2 {channelIndex = 0 : i32}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @y_out({{.*}}) drains @y_l2 {channelIndex = 0 : i32}
// CHECK-DAG:   aie.route_endpoint @x_shim({{.*}}) DMA {channelIndex = 1 : i32, fifoName = "x"}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @x_in({{.*}}) fills @x_l2 {channelIndex = 1 : i32}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @x_out({{.*}}) drains @x_l2 {channelIndex = 1 : i32}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @w_core({{.*}}) drains @w_l1 {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
// CHECK-DAG:   aie.route_endpoint @w_shim({{.*}}) DMA {channelIndex = 0 : i32, fifoName = "w"}
// CHECK-DAG:   aie.objectfifo.dma_endpoint @v_core({{.*}}) drains @v_l1 {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
// CHECK-DAG:   aie.route_endpoint @v_shim({{.*}}) DMA {channelIndex = 1 : i32, fifoName = "v"}

module @declared {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem = aie.tile(0, 1)
    %core2 = aie.tile(0, 2)
    %core3 = aie.tile(0, 3)

    aie.objectfifo.pool @a_l2(%mem) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 64 : i32}
    }
    aie.objectfifo.pool @a_l1(%core2) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 64 : i32}
    }
    aie.route_endpoint @a_shim(%shim) DMA {fifoName = "a"}
    aie.objectfifo.dma_endpoint @a_in(%mem) fills @a_l2
    aie.objectfifo.dma_endpoint @a_out(%mem) drains @a_l2
    aie.objectfifo.dma_endpoint @a_core(%core2) fills @a_l1
    aie.objectfifo.core_endpoint @a_use(%core2) drains @a_l1
    aie.route from @a_shim to [@a_in]
    aie.route from @a_out to [@a_core]

    aie.objectfifo.pool @b_l2(%mem) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 64 : i32}
    }
    aie.objectfifo.pool @b_l1(%core3) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 64 : i32}
    }
    aie.route_endpoint @b_shim(%shim) DMA {fifoName = "b"}
    aie.objectfifo.dma_endpoint @b_in(%mem) fills @b_l2
    aie.objectfifo.dma_endpoint @b_out(%mem) drains @b_l2
    aie.objectfifo.dma_endpoint @b_core(%core3) fills @b_l1
    aie.objectfifo.core_endpoint @b_use(%core3) drains @b_l1
    aie.route from @b_shim to [@b_in]
    aie.route from @b_out to [@b_core]

    aie.objectfifo.pool @c_l1(%core2) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @c_make(%core2) fills @c_l1
    aie.objectfifo.dma_endpoint @c_core(%core2) drains @c_l1
    aie.route_endpoint @c_shim(%shim) DMA {fifoName = "c"}
    aie.route from @c_core to [@c_shim] {packet}

    aie.objectfifo.pool @d_l1(%core3) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @d_make(%core3) fills @d_l1
    aie.objectfifo.dma_endpoint @d_core(%core3) drains @d_l1
    aie.route_endpoint @d_shim(%shim) DMA {fifoName = "d"}
    aie.route from @d_core to [@d_shim] {packet}
  }
}

// -----

module @permuted {
  aie.device(npu2) {
    %core3 = aie.tile(0, 3)
    %core2 = aie.tile(0, 2)
    %mem = aie.tile(0, 1)
    %shim = aie.tile(0, 0)

    aie.objectfifo.pool @v_l1(%core3) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @t {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @v_make(%core3) fills @v_l1
    aie.route_endpoint @v_shim(%shim) DMA {fifoName = "v"}
    aie.objectfifo.dma_endpoint @v_core(%core3) drains @v_l1
    aie.route from @v_core to [@v_shim] {packet}

    aie.objectfifo.pool @x_l1(%core3) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @t {offset = 0 : i32, size = 64 : i32}
    }
    aie.objectfifo.pool @x_l2(%mem) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @t {offset = 0 : i32, size = 64 : i32}
    }
    aie.objectfifo.dma_endpoint @x_out(%mem) drains @x_l2
    aie.objectfifo.dma_endpoint @x_core(%core3) fills @x_l1
    aie.objectfifo.core_endpoint @x_use(%core3) drains @x_l1
    aie.route from @x_out to [@x_core]
    aie.objectfifo.dma_endpoint @x_in(%mem) fills @x_l2
    aie.route_endpoint @x_shim(%shim) DMA {fifoName = "x"}
    aie.route from @x_shim to [@x_in]

    aie.objectfifo.pool @w_l1(%core2) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @t {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @w_core(%core2) drains @w_l1
    aie.objectfifo.core_endpoint @w_make(%core2) fills @w_l1
    aie.route_endpoint @w_shim(%shim) DMA {fifoName = "w"}
    aie.route from @w_core to [@w_shim] {packet}

    aie.objectfifo.pool @y_l1(%core2) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @t {offset = 0 : i32, size = 64 : i32}
    }
    aie.objectfifo.pool @y_l2(%mem) {depth = 2 : i32} : memref<64xi32> {
      aie.objectfifo.segment @t {offset = 0 : i32, size = 64 : i32}
    }
    aie.objectfifo.dma_endpoint @y_core(%core2) fills @y_l1
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y_l2
    aie.objectfifo.core_endpoint @y_use(%core2) drains @y_l1
    aie.objectfifo.dma_endpoint @y_in(%mem) fills @y_l2
    aie.route_endpoint @y_shim(%shim) DMA {fifoName = "y"}
    aie.route from @y_out to [@y_core]
    aie.route from @y_shim to [@y_in]
  }
}

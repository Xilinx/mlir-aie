//===- shared_channel_selects_out_of_order.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A channel its fifos receive in turn holds their order, so they cannot share
// it. Only an out-of-order channel holds none, each object naming the
// descriptor that takes it. Pinning two fifos to one channel therefore asks
// for an out-of-order channel, and selects it.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// CHECK-LABEL: @shared_selects_out_of_order
// CHECK:       aie.objectfifo.dma_endpoint @a_cons_dma{{.*}}out_of_order
// CHECK:       aie.objectfifo.dma_endpoint @b_cons_dma{{.*}}out_of_order
module @shared_selects_out_of_order {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, out_of_order, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Each object names its descriptor in the packet header, so a fifo sharing a
// channel needs a packet route to carry one.
module @shared_needs_packet {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, out_of_order, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares a receiving channel, which it can only do out of order}}
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
  }
}

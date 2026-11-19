//===- shared_channel_requires_out_of_order.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Receiving channel sharing requires explicit out_of_order on every FIFO.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// CHECK-LABEL: @shared_explicit_out_of_order
// CHECK:       aie.objectfifo.dma_endpoint @a__cons__dma{{.*}}out_of_order
// CHECK:       aie.objectfifo.dma_endpoint @b__cons__dma{{.*}}out_of_order
module @shared_explicit_out_of_order {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
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
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares a receiving channel; explicitly set out_of_order = true}}
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

module @omitted {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares a receiving channel; explicitly set out_of_order = true}}
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
  }
}


// -----

module @false {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @a(%tile_0_2, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares a receiving channel; explicitly set out_of_order = true}}
    aie.objectfifo @b(%tile_0_3, {%tile_0_5}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>, out_of_order = false>, cons_ports = [#aie.end_port<DMA : 1>]} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// CHECK-LABEL: module @auto_shared_memory
// CHECK: aie.objectfifo.pool @a
// CHECK: aie.objectfifo.pool @b
// CHECK-NOT: aie.objectfifo.dma_endpoint
module @auto_shared_memory {
  aie.device(npu2) {
    %producer = aie.tile(0, 2)
    %consumer = aie.tile(0, 3)
    aie.objectfifo @a(%producer, {%consumer}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%producer, {%consumer}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// CHECK-LABEL: module @explicit_shared_memory
// CHECK: aie.objectfifo.pool @a
// CHECK: aie.objectfifo.pool @b
// CHECK-NOT: aie.objectfifo.dma_endpoint
module @explicit_shared_memory {
  aie.device(npu2) {
    %producer = aie.tile(0, 2)
    %consumer = aie.tile(0, 3)
    aie.objectfifo @a(%producer, {%consumer}, 2 : i32) {transport = #aie.transport<shared_mem>, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%producer, {%consumer}, 2 : i32) {transport = #aie.transport<shared_mem>, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// A shim's receiving channel takes no turns by header: allocation shares no
// pinned S2MM there, out_of_order or not, and the split says so first.
module @shim_receive_shared {
  aie.device(npu2) {
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    %s = aie.tile(0, 0)
    // expected-error@+1 {{shares a receiving channel, but a shim's receiving channel writes into the runtime's tasks, which no object header picks}}
    aie.objectfifo @fa(%a, {%s}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fb(%b, {%s}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
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
  }
}

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// Three remote compute producers use one shared receiving channel.
// CHECK: aie.tile({{[0-9]+}}, 5)
// CHECK-DAG: aie.objectfifo.dma_endpoint @a__cons__dma{{.*}}channelIndex = 0
// CHECK-DAG: aie.objectfifo.dma_endpoint @b__cons__dma{{.*}}channelIndex = 0
// CHECK-DAG: aie.objectfifo.dma_endpoint @c__cons__dma{{.*}}channelIndex = 0
module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, 5)
    %logical_core_0 = aie.logical_tile<CoreTile>(0, 2)
    aie.objectfifo @a(%logical_core_0, {%logical_core}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %logical_core_1 = aie.logical_tile<CoreTile>(1, 2)
    aie.objectfifo @b(%logical_core_1, {%logical_core}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    %logical_core_2 = aie.logical_tile<CoreTile>(2, 2)
    aie.objectfifo @c(%logical_core_2, {%logical_core}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
  }
}


// -----

// Sharing an S2MM channel does not reduce the sender's MM2S demand.
module {
  aie.device(npu2) {
    %logical_core = aie.logical_tile<CoreTile>(?, 5)
    // expected-error@+1 {{requires 0 input/3 output DMA channels}}
    %logical_core_0 = aie.logical_tile<CoreTile>(0, 2)
    aie.objectfifo @a(%logical_core_0, {%logical_core}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%logical_core_0, {%logical_core}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%logical_core_0, {%logical_core}, 2 : i32) {cons_ports = [#aie.end_port<DMA : 0>], transport = #aie.transport<dma, packet = #aie.packet_info<>, out_of_order = true>} : !aie.objectfifo<memref<16xi32>>
  }
}

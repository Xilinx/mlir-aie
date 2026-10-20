//===- objectfifo_transport.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// Every transport mode round-trips, the DMA path carries a packet header, open
// or pinned, and each end names what drives it.

// CHECK-LABEL: @transport_modes
// CHECK: aie.objectfifo @auto_path
// CHECK-NOT: transport
// CHECK: aie.objectfifo @dma_path{{.*}}transport = #aie.transport<dma>
// CHECK: aie.objectfifo @shared_path{{.*}}transport = #aie.transport<shared_mem>
// CHECK: aie.objectfifo @packet_path{{.*}}transport = #aie.transport<dma, packet = #aie.packet_info<>>
// CHECK: aie.objectfifo @pinned_path{{.*}}transport = #aie.transport<auto, packet = #aie.packet_info<pkt_id = 7>>
module @transport_modes {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @auto_path (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @dma_path (%tile12, {%tile22}, 2 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @shared_path (%tile22, {%tile23}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @packet_path (%tile12, {%tile23}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<>>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @pinned_path (%tile13, {%tile22}, 2 : i32) {transport = #aie.transport<auto, packet = #aie.packet_info<pkt_id = 7>>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

// CHECK-LABEL: @end_ports
// CHECK: aie.objectfifo @stream_path{{.*}}cons_ports = [#aie.end_port<Core : 1>], prod_port = #aie.end_port<Core : 1>
// CHECK: aie.objectfifo @pinned_dma{{.*}}cons_ports = [#aie.end_port<DMA>, #aie.end_port<DMA : 0>], prod_port = #aie.end_port<DMA : 1>
module @end_ports {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @stream_path (%tile13, {%tile23}, 2 : i32) {prod_port = #aie.end_port<Core : 1>, cons_ports = [#aie.end_port<Core : 1>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @pinned_dma (%tile12, {%tile22, %tile23}, 2 : i32) {prod_port = #aie.end_port<DMA : 1>, cons_ports = [#aie.end_port<DMA>, #aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
 }
}

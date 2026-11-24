//===- ctrl_pkt_overlay_pinned.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The control overlay sits on the shim MM2S channel
// aie-generate-column-control-overlay picks, and shares it with the packet
// ends pinned there when they are never in flight with its control packets.
// Both placers see it share channel 0 with @a, as the overlay does, rather
// than count a third channel; before, the sequential placer refused ("1/2
// output" used) and the SA warned.

// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true})' %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true ctrlpkt-adaptive-overlay=false})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=1})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true placer=sa_placer sa-seed=2})' %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true ctrlpkt-adaptive-overlay=false placer=sa_placer sa-seed=3})' %s | FileCheck %s

// CHECK-LABEL: module @overlay_pinned_packets
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[S:.*]], MM2S, 0) {aie.is_ctrl_pkt_overlay, aie.shareable_with = [@a__shim_alloc, @b__shim_alloc]}
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%[[S]], MM2S, 0, <pkt_id = 3>)
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%[[S]], MM2S, 1, <pkt_id = 4>)
module @overlay_pinned_packets {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%c}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
    }
  }
}

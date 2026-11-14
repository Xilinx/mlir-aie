//===- ctrl_pkt_overlay_packet_sw.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ctrl_pkt_overlay_pinned.mlir's design with no packet headers written: under
// packet-sw-objFifos its fifos are packet switched all the same, so the
// overlay shares channel 0 with them as it does there, and as the placers
// expect.

// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true ctrlpkt-adaptive-overlay=false packet-sw-objFifos=true})' %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages{generate-ctrl-pkt-overlay=true ctrlpkt-adaptive-overlay=false packet-sw-objFifos=true placer=sa_placer sa-seed=1})' %s | FileCheck %s

// CHECK-LABEL: module @overlay_packet_sw
// CHECK: aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(%[[S:.*]], MM2S, 0) {aie.is_ctrl_pkt_overlay, aie.shareable_with = [@a__shim_alloc, @b__shim_alloc]}
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%[[S]], MM2S, 0, <pkt_id = {{[0-9]+}}>)
// CHECK-DAG: aie.shim_dma_allocation @b__shim_alloc(%[[S]], MM2S, 1, <pkt_id = {{[0-9]+}}>)
module @overlay_packet_sw {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 1>} : !aie.objectfifo<memref<16xi32>>
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

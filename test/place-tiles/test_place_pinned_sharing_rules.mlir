//===- test_place_pinned_sharing_rules.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two ends pinning one channel share it only where allocation and the split
// let them: a receiving end only off a shim, and a core's sending ends only
// where each can push its tasks. Both placers refuse with their reason
// rather than placing what lowering then refuses.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' %s

module @shim_receive_ooo {
  aie.device(npu2) {
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @fa(%a, {%s}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares S2MM channel 0 with @fa, but a shim's receiving channel writes into the runtime's tasks, which no object header picks}}
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

// -----

module @core_send_init {
  aie.device(npu2) {
    %core_tile = aie.logical_tile<CoreTile>(?, ?)
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(1, 0)
    aie.objectfifo @a(%core_tile, {%s0}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>> = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]
    // expected-error@+1 {{shares MM2S channel 0 with @a, but @a's push_queue does not support initialized objects}}
    aie.objectfifo @b(%core_tile, {%s1}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%core_tile) {
      %a = aie.objectfifo.acquire @a(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @a(Produce, 1)
      %b = aie.objectfifo.acquire @b(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @b(Produce, 1)
      aie.end
    }
  }
}

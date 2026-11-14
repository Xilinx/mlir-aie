//===- test_sa_pinned_bd_ids.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The SA placer weighs a mem tile's BD ids with each end on the channel it
// pins, ends pinning one channel together, as objectFIFO allocation does.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=2' %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=3' %s | FileCheck %s

// Two out-of-order fifos share pinned S2MM 0 with 14 BDs each, 28 of the 24
// BD ids an even channel reaches; neither pool can be walked.
// CHECK-LABEL: module @shared_out_of_order
module @shared_out_of_order {
  // expected-warning@+1 {{SA placer's resource estimate is over capacity (penalty=20)}}
  aie.device(npu2) {
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    // expected-note-re@+1 {{BD ids penalty=20 on ({{[0-9]+}}, 1)}}
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @fa(%a, {%m}, [2, 14]) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fb(%b, {%m}, [2, 14]) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @fa(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @fb(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fb(Produce, 1)
      aie.end
    }
  }
}

// -----

// Twelve BDs each fit.
// CHECK-LABEL: module @shared_fits
module @shared_fits {
  aie.device(npu2) {
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @fa(%a, {%m}, [2, 12]) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fb(%b, {%m}, [2, 12]) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @fa(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @fb(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fb(Produce, 1)
      aie.end
    }
  }
}

// -----

// Pinned S2MM 0 and 2 reach the same BD ids; walking one pool fits them.
// CHECK-LABEL: module @even_channels
module @even_channels {
  aie.device(npu2) {
    %a = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.logical_tile<CoreTile>(?, ?)
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @fa(%a, {%m}, [2, 14]) {cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @fb(%b, {%m}, [2, 14]) {cons_ports = [#aie.end_port<DMA : 2>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @fa(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @fb(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fb(Produce, 1)
      aie.end
    }
  }
}

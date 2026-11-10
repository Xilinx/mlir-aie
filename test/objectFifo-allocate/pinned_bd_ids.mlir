//===- pinned_bd_ids.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ends pinned to a mem tile's channels must fit the BD ids those channels
// reach, ends sharing a pinned channel together, as unpinned ends must: a
// pool that can be walked whole is, and one that cannot is reported here
// rather than by aie-assign-bd-ids. Even channels reach BD ids 0-23.

// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --verify-diagnostics %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate --aie-objectfifo-lower-dmas --aie-objectfifo-lower-cores --aie-objectfifo-erase-pools --aie-assign-lock-ids --aie-assign-bd-ids --verify-diagnostics %s -o /dev/null

// Two out-of-order fifos share pinned S2MM 0 with 14 BDs each, and neither
// pool can be walked.
module @shared_out_of_order {
  // expected-remark@+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %a = aie.tile(0, 2)
    %b = aie.tile(1, 2)
    %m = aie.tile(0, 1)
    // expected-error@+1 {{needs 14 buffer descriptors, but pinned S2MM DMA channel 0 of its tile has 10 left beside the ends on the channels sharing its BD ids}}
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

// One out-of-order fifo pins S2MM 0 with 28 BDs.
module @single_out_of_order {
  // expected-remark@+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    %a = aie.tile(0, 2)
    %m = aie.tile(0, 1)
    // expected-error@+1 {{needs 28 buffer descriptors, but pinned S2MM DMA channel 0 of its tile has 24 left beside the ends on the channels sharing its BD ids}}
    aie.objectfifo @fa(%a, {%m}, [2, 28]) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>, out_of_order = true>, iter_count = 1 : i32, cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @fa(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Produce, 1)
      aie.end
    }
  }
}

// -----

// The same end in order is walked whole, one BD id a side.
// CHECK-LABEL: module @single_walked
// CHECK: aie.objectfifo.pool @fa__cons__pool(%{{.*}}) {buffers = [@fa__cons__buf0], contiguous, depth = 28
// CHECK: aie.objectfifo.dma_endpoint @fa__cons__dma(%{{.*}}) fills @fa__cons__pool {channelIndex = 0
module @single_walked {
  aie.device(npu2) {
    %a = aie.tile(0, 2)
    %m = aie.tile(0, 1)
    aie.objectfifo @fa(%a, {%m}, [2, 28]) {cons_ports = [#aie.end_port<DMA : 0>]} : !aie.objectfifo<memref<16xi32>>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @fa(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @fa(Produce, 1)
      aie.end
    }
  }
}

// -----

// Pinned S2MM 0 and 2 reach the same BD ids, so one of the two pools is walked.
// CHECK-LABEL: module @even_channels
// CHECK: aie.objectfifo.pool @fa__cons__pool(%{{.*}}) {buffers = [@fa__cons__buf0], contiguous, depth = 14
module @even_channels {
  aie.device(npu2) {
    %a = aie.tile(0, 2)
    %b = aie.tile(1, 2)
    %m = aie.tile(0, 1)
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

// -----

// Twelve BDs each fit one even channel's 24 together.
// CHECK-LABEL: module @shared_fits
// CHECK: aie.objectfifo.dma_endpoint @fa__cons__dma(%{{.*}}) fills @fa__cons__pool {channelIndex = 0
// CHECK: aie.objectfifo.dma_endpoint @fb__cons__dma(%{{.*}}) fills @fb__cons__pool {channelIndex = 0
module @shared_fits {
  aie.device(npu2) {
    %a = aie.tile(0, 2)
    %b = aie.tile(1, 2)
    %m = aie.tile(0, 1)
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

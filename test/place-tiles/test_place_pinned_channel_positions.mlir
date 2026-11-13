//===- test_place_pinned_channel_positions.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ends of different logical tiles that pin one DMA channel meet only where the
// placers put them together. Both placers keep two that cannot share it apart
// and count two that can as one channel, as objectFIFO allocation does.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// Two logical shims each send a circuit fifo on pinned MM2S 0: they take two
// shim tiles. Before, the sequential placer put both on (0, 0).
// CHECK-LABEL: module @two_shims_same_pin
// CHECK: aie.route_endpoint @a__prod__dma(%[[A:[a-z_0-9]+]]) DMA {{.*}}channelIndex = 0
// CHECK-NOT: aie.route_endpoint @b__prod__dma(%[[A]])
// CHECK: aie.route_endpoint @b__prod__dma({{.*}}channelIndex = 0
module @two_shims_same_pin {
  aie.device(npu2) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
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

// -----

// The same on two columns with a third shim. Before, the SA put both on one
// shim on seeds 1-6.
// CHECK-LABEL: module @two_shims_same_pin_2col
// CHECK: aie.route_endpoint @a__prod__dma(%[[A:[a-z_0-9]+]]) DMA {{.*}}channelIndex = 0
// CHECK-NOT: aie.route_endpoint @b__prod__dma(%[[A]])
// CHECK: aie.route_endpoint @b__prod__dma({{.*}}channelIndex = 0
module @two_shims_same_pin_2col {
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c}, 2 : i32) {prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d(%s2, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%c2) {
      %d = aie.objectfifo.acquire @d(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @d, id = 2 : i64} : memref<16xi32>
    }
  }
}

// -----

// Two logical shims on one column send packet fifos with distinct ids on
// pinned MM2S 0: placed together they share the channel. Before, the
// sequential placer counted it twice and refused, and the SA warned.
// CHECK-LABEL: module @s3_two_logical_shims_samepin_pkt
// CHECK: aie.route_endpoint @a__prod__dma(%[[S:[a-z_0-9]+]]) DMA {{.*}}channelIndex = 0{{.*}}pkt_id = 3
// CHECK: aie.route_endpoint @b__prod__dma(%[[S]]) DMA {{.*}}channelIndex = 0{{.*}}pkt_id = 4
// CHECK: aie.route_endpoint @d__prod__dma(%[[S]]) DMA {{.*}}channelIndex = 1
module @s3_two_logical_shims_samepin_pkt {
  aie.device(npu2_1col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%s0, {%c0}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c1}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d(%s1, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%c1) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%c2) {
      %d = aie.objectfifo.acquire @d(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @d, id = 2 : i64} : memref<16xi32>
    }
  }
}

// -----

// One of those ends on a physical shim, the other on a logical shim placed on
// it.
// CHECK-LABEL: module @s2b_phys_and_logical_samepin_pkt
// CHECK: aie.route_endpoint @a__prod__dma(%[[S:[a-z_0-9]+]]) DMA {{.*}}channelIndex = 0{{.*}}pkt_id = 3
// CHECK: aie.route_endpoint @b__prod__dma(%[[S]]) DMA {{.*}}channelIndex = 0{{.*}}pkt_id = 4
// CHECK: aie.route_endpoint @d__prod__dma(%[[S]]) DMA {{.*}}channelIndex = 1
module @s2b_phys_and_logical_samepin_pkt {
  aie.device(npu2_1col) {
    %p = aie.tile(0, 0)
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a(%p, {%c0}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s, {%c1}, 2 : i32) {transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 4>>, prod_port = #aie.end_port<DMA : 0>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d(%s, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c0) {
      %a = aie.objectfifo.acquire @a(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @a(Consume, 1)
      aie.end
    }
    aie.core(%c1) {
      %b = aie.objectfifo.acquire @b(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @b(Consume, 1)
      aie.end
    }
    aie.core(%c2) {
      %d = aie.objectfifo.acquire @d(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @d(Consume, 1)
      aie.end
    }
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @b, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @d, id = 2 : i64} : memref<16xi32>
    }
  }
}

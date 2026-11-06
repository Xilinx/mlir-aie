// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-bd-ids --split-input-file --verify-diagnostics %s | FileCheck %s

// Explicit receiving IDs for one broadcast slot must agree.
module @mismatch {
  aie.device(npu2) {
    %x = aie.tile(0, 2)
    %y = aie.tile(1, 2)
    %s = aie.tile(2, 2)
    %bx = aie.buffer(%x) : memref<8xi32>
    %by = aie.buffer(%y) : memref<8xi32>
    %bs = aie.buffer(%s) : memref<8xi32>
    aie.packet_flow(0) {
      aie.packet_source<%s, DMA : 0>
      aie.packet_dest<%x, DMA : 0>
      aie.packet_dest<%y, DMA : 0>
    } {keep_pkt_header = true}
    aie.mem(%x) {
      aie.dma_start(S2MM, 0, ^bd, ^end) {out_of_order}
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%bx : memref<8xi32>) {bd_id = 0 : i32, bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%y) {
      aie.dma_start(S2MM, 0, ^bd, ^end) {out_of_order}
    ^bd:
      aie.dma_bd_packet(0, 0)
      // expected-error@+1 {{broadcast receiving buffers must have matching bd_id values for bd_tag 7}}
      aie.dma_bd(%by : memref<8xi32>) {bd_id = 1 : i32, bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%s) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%bs : memref<8xi32>) {out_of_order_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// Reserve a common ID before ordinary BDs occupy it on either destination.
// CHECK-LABEL: module @allocate
// CHECK: aie.dma_bd(%{{.*}}) {bd_id = 1 : i32}
// CHECK: aie.dma_bd(%{{.*}}) {bd_id = 1 : i32}
// CHECK: aie.dma_bd(%{{.*}}) {bd_id = 0 : i32, out_of_order_id = 1 : i32}
module @allocate {
  aie.device(npu2) {
    %x = aie.tile(0, 2) {aiex.reserved_bd_ids = array<i32: 0>}
    %y = aie.tile(1, 2)
    %s = aie.tile(2, 2)
    %bx = aie.buffer(%x) : memref<8xi32>
    %by = aie.buffer(%y) : memref<8xi32>
    %bs = aie.buffer(%s) : memref<8xi32>
    aie.packet_flow(0) {
      aie.packet_source<%s, DMA : 0>
      aie.packet_dest<%x, DMA : 0>
      aie.packet_dest<%y, DMA : 0>
    } {keep_pkt_header = true}
    aie.mem(%x) {
      aie.dma_start(S2MM, 0, ^bd, ^end) {out_of_order}
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%bx : memref<8xi32>) {bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%y) {
      aie.dma_start(S2MM, 0, ^bd, ^end) {out_of_order}
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%by : memref<8xi32>) {bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%s) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%bs : memref<8xi32>) {out_of_order_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

// A pinned ID on one receiver constrains allocation on all destinations.
// CHECK-LABEL: module @pinned
// CHECK: aie.dma_bd(%{{.*}}) {bd_id = 3 : i32}
// CHECK: aie.dma_bd(%{{.*}}) {bd_id = 3 : i32}
// CHECK: aie.dma_bd(%{{.*}}) {bd_id = 0 : i32, out_of_order_id = 3 : i32}
module @pinned {
  aie.device(npu2) {
    %x = aie.tile(0, 2) {aiex.reserved_bd_ids = array<i32: 0>}
    %y = aie.tile(1, 2)
    %s = aie.tile(2, 2)
    %bx = aie.buffer(%x) : memref<8xi32>
    %by = aie.buffer(%y) : memref<8xi32>
    %bs = aie.buffer(%s) : memref<8xi32>
    aie.packet_flow(0) {
      aie.packet_source<%s, DMA : 0>
      aie.packet_dest<%x, DMA : 0>
      aie.packet_dest<%y, DMA : 0>
    } {keep_pkt_header = true}
    aie.mem(%x) {
      aie.dma_start(S2MM, 0, ^bd, ^end) {out_of_order}
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%bx : memref<8xi32>) {bd_id = 3 : i32, bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%y) {
      aie.dma_start(S2MM, 0, ^bd, ^end) {out_of_order}
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%by : memref<8xi32>) {bd_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
    aie.mem(%s) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%bs : memref<8xi32>) {out_of_order_tag = 7 : i32}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

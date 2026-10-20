//===- test_place_shared_shim_packet_channel.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Producer endpoints that pin the same shim MM2S channel share it. The runtime
// writes a shim channel's descriptors one transfer at a time, each carrying
// its own packet header, so the tile pays for the channel once and the
// switchbox tells the streams apart by id.
//
// Three fifos leave one shim tile that has two MM2S channels. Without sharing
// the third has nowhere to go.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles %s | FileCheck %s

// CHECK-LABEL: @shared_shim_channel
// CHECK:       %[[S:.*]] = aie.tile(0, 0)
// CHECK-DAG:   aie.objectfifo @a(%[[S]],
// CHECK-DAG:   aie.objectfifo @b(%[[S]],
// CHECK-DAG:   aie.objectfifo @c(%[[S]],
module @shared_shim_channel {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    aie.objectfifo @a(%shim, {%c0}, 2 : i32) {packet, prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%shim, {%c1}, 2 : i32) {packet, prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%shim, {%c2}, 2 : i32) {packet, prod_dma_channel = 1 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Each endpoint pinned to a tile gets a logical tile of its own, which is what
// IRON's prod(tile=...) builds. They share the channel all the same.
// CHECK-LABEL: @shared_across_logical_tiles
// CHECK:       %[[S:.*]] = aie.tile(0, 0)
// CHECK-DAG:   aie.objectfifo @a(%[[S]],
// CHECK-DAG:   aie.objectfifo @b(%[[S]],
// CHECK-DAG:   aie.objectfifo @c(%[[S]],
// CHECK-DAG:   aie.objectfifo @d(%[[S]],
module @shared_across_logical_tiles {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(0, 4)
    %c3 = aie.logical_tile<CoreTile>(0, 5)
    %s0 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s1 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s2 = aie.logical_tile<ShimNOCTile>(0, 0)
    %s3 = aie.logical_tile<ShimNOCTile>(0, 0)
    aie.objectfifo @a(%s0, {%c0}, 2 : i32) {packet, packet_id = 12 : i8, prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%s1, {%c1}, 2 : i32) {packet, packet_id = 13 : i8, prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c(%s2, {%c2}, 2 : i32) {packet, packet_id = 12 : i8, prod_dma_channel = 1 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d(%s3, {%c3}, 2 : i32) {packet, packet_id = 13 : i8, prod_dma_channel = 1 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// A circuit-switched fifo cannot share: nothing separates the two streams.
module @shared_needs_packet {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    aie.objectfifo @a(%shim, {%c0}, 2 : i32) {packet, prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares MM2S channel 0 with @a, but a shared channel carries both at once, which only packet-switched routes tell apart}}
    aie.objectfifo @b(%shim, {%c1}, 2 : i32) {prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Two endpoints that pin the same id are indistinguishable on the wire.
module @shared_needs_distinct_ids {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    aie.objectfifo @a(%shim, {%c0}, 2 : i32) {packet, packet_id = 4 : i8, prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    // expected-error@+1 {{shares MM2S channel 0 with @a, but both pin packet id 4}}
    aie.objectfifo @b(%shim, {%c1}, 2 : i32) {packet, packet_id = 4 : i8, prod_dma_channel = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}

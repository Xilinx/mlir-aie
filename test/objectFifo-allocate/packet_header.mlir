//===- packet_header.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-allocate %s | FileCheck %s

// A route asks for packet switching with the same `#aie.packet_info` its
// endpoints and buffer descriptors carry afterwards; allocation only fills in
// the id. A pinned id and a non-default type both survive, and an open id is
// assigned around the pinned ones.

module @packet_header {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)
    %tile32 = aie.tile(3, 2)

    aie.objectfifo.pool @a_pool(%tile12) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @a_prod(%tile12) drains @a_pool
    aie.objectfifo.pool @a_cons_pool(%tile33) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @a_cons(%tile33) fills @a_cons_pool

    aie.objectfifo.pool @b_pool(%tile13) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @b_prod(%tile13) drains @b_pool
    aie.objectfifo.pool @b_cons_pool(%tile32) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @b_cons(%tile32) fills @b_cons_pool

    aie.route from @a_prod to [@a_cons] {packet = #aie.packet_info<pkt_type = 1, pkt_id = 0>}
    aie.route from @b_prod to [@b_cons] {packet = #aie.packet_info<>}
  }
}

// CHECK-LABEL: @packet_header
// The pinned header is stamped on the source as written, type included.
// CHECK:       aie.objectfifo.dma_endpoint @a_prod({{.*}}) drains @a_pool {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_type = 1, pkt_id = 0>}
// The open header gets the next free id, skipping the pinned 0.
// CHECK:       aie.objectfifo.dma_endpoint @b_prod({{.*}}) drains @b_pool {channelIndex = 0 : i32, packet = #aie.packet_info<pkt_id = 1>}
// CHECK:       aie.packet_flow(0) {
// CHECK:       aie.packet_flow(1) {
// CHECK-NOT:   aie.route

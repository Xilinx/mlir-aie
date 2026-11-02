//===- dispatch_bad.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectfifo-verify --verify-diagnostics %s

// Every route from a dispatching endpoint carries the header that tells its
// turn from the others.
module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32, initValues = [dense<0> : memref<16xi32>, dense<0> : memref<16xi32>]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a_in, @b_in]}
    aie.route_endpoint @a_in(%a) Core {channelIndex = 0 : i32}
    aie.route_endpoint @b_in(%b) Core {channelIndex = 0 : i32}
    // expected-error@+1 {{leaves dispatching endpoint @y_out and needs a packet header to tell its turns apart}}
    aie.route from @y_out to [@a_in]
    aie.route from @y_out to [@b_in] {packet = #aie.packet_info<>}
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32, initValues = [dense<0> : memref<16xi32>, dense<0> : memref<16xi32>]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a_in]}
    aie.route_endpoint @a_in(%a) Core {channelIndex = 0 : i32}
    aie.route_endpoint @b_in(%b) Core {channelIndex = 0 : i32}
    aie.route from @y_out to [@a_in] {packet = #aie.packet_info<>}
    // expected-error@+1 {{reaches @b_in, which dispatching endpoint @y_out never takes a turn for}}
    aie.route from @y_out to [@b_in] {packet = #aie.packet_info<>}
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32, initValues = [dense<0> : memref<16xi32>, dense<0> : memref<16xi32>]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{dispatches to @b_in, but no route from it reaches there}}
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a_in, @b_in]}
    aie.route_endpoint @a_in(%a) Core {channelIndex = 0 : i32}
    aie.route_endpoint @b_in(%b) Core {channelIndex = 0 : i32}
    aie.route from @y_out to [@a_in] {packet = #aie.packet_info<>}
  }
}

// -----

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32, initValues = [dense<0> : memref<16xi32>, dense<0> : memref<16xi32>]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a_in, @b_in]}
    aie.route_endpoint @a_in(%a) Core {channelIndex = 0 : i32}
    aie.route_endpoint @b_in(%b) Core {channelIndex = 0 : i32}
    // expected-error@+1 {{leaves dispatching endpoint @y_out for one destination, so it has one source and one destination}}
    aie.route from @y_out to [@a_in, @b_in] {packet = #aie.packet_info<>}
  }
}

// -----

// A time-mode dispatch's turns move the same chain through one shared
// endpoint, one object per turn, so a turn whose consumer takes fewer
// elements than the others would under- or over-drain the shared pool.
module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %a = aie.tile(0, 2)
    %b = aie.tile(0, 3)
    aie.objectfifo.pool @y(%mem) {depth = 2 : i32, initValues = [dense<0> : memref<16xi32>, dense<0> : memref<16xi32>]} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{dispatches, so every turn moves 16 elements, but @a_dma moves 4}}
    aie.objectfifo.dma_endpoint @y_out(%mem) drains @y {dispatch = [@a_dma, @b_dma]}
    aie.objectfifo.pool @a_pool(%a) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @a_dma(%a) fills @a_pool {dimensions = #aie<bd_dim_layout_array_array[[<size = 4, stride = 1>]]>}
    aie.objectfifo.core_endpoint @a_cons(%a) drains @a_pool
    aie.objectfifo.pool @b_pool(%b) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @b_dma(%b) fills @b_pool
    aie.objectfifo.core_endpoint @b_cons(%b) drains @b_pool
    aie.route from @y_out to [@a_dma] {packet = #aie.packet_info<>}
    aie.route from @y_out to [@b_dma] {packet = #aie.packet_info<pkt_id = 4>}
  }
}

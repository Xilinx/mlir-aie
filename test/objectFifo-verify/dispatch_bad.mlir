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

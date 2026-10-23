//===- bad_packet_unassigned.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A packet header without an id is a request for allocation to pick one, so
// it belongs on `aie.route`. Everything below that level carries the header
// allocation resolved, and rejects one with the id still open.

module {
  aie.device(xcve2302) {
    %tile00 = aie.tile(0, 0)
    // expected-error@+1 {{'aie.route_endpoint' op packet has no pkt_id; allocation assigns one, so a header at this level must carry it}}
    aie.route_endpoint @in(%tile00) DMA {packet = #aie.packet_info<>}
  }
}

// -----

module {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    aie.objectfifo.pool @pool(%tile12) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    // expected-error@+1 {{'aie.objectfifo.dma_endpoint' op packet has no pkt_id}}
    aie.objectfifo.dma_endpoint @prod(%tile12) drains @pool {packet = #aie.packet_info<pkt_type = 1>}
  }
}

// -----

module {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %buf = aie.buffer(%tile12) : memref<16xi32>
    %mem = aie.mem(%tile12) {
      aie.dma_start(MM2S, 0, ^bd, ^end)
    ^bd:
      // expected-error@+1 {{'aie.dma_bd' op packet has no pkt_id}}
      aie.dma_bd(%buf : memref<16xi32>) {packet = #aie.packet_info<>}
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}

// -----

module {
  aie.device(xcve2302) {
    %tile00 = aie.tile(0, 0)
    aie.shim_dma_allocation @of(%tile00, MM2S, 0, <pkt_id = 3>)
    // expected-error@+1 {{'aie.shim_dma_allocation' op packet has no pkt_id}}
    aie.shim_dma_allocation @of2(%tile00, MM2S, 1, <>)
  }
}

// -----

module {
  aie.device(xcve2302) {
    // expected-error@+1 {{'aie.tile' op controller_id has no pkt_id}}
    %tile00 = aie.tile(0, 0) {controller_id = #aie.packet_info<>}
  }
}

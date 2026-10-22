// RUN: aie-opt --aie-objectfifo-verify --split-input-file --verify-diagnostics %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Packet routes may share an aie.route_endpoint, whose own verifier says
// which may.
module @shared_route_endpoint {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
    aie.route_endpoint @host(%shim) DMA
    aie.route_endpoint @a(%mt) DMA
    aie.route_endpoint @b(%mt) DMA
    aie.route from @a to [@host] {packet, packet_id = 1 : i8}
    aie.route from @b to [@host] {packet, packet_id = 2 : i8}
  }
}

// -----

// An objectfifo DMA endpoint drives one channel, so two flows may not name it.
module @shared_dma_endpoint {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)
    aie.objectfifo.pool @p(%tile12) {
      depth = 2 : i32
    } : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @c(%tile12) fills @p
    // expected-error@+1 {{drives one channel, so at most one flow may name it, but it is named 2 times}}
    aie.objectfifo.dma_endpoint @d(%tile12) drains @p
    aie.route_endpoint @x(%tile33) DMA
    aie.route_endpoint @y(%tile33) DMA
    aie.route from @d to [@x]
    aie.route from @d to [@y]
  }
}

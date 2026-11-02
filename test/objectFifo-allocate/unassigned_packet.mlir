//===- unassigned.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --aie-objectfifo-allocate %s

// Allocation lowers a packet header; it no longer picks its id.

module {
  aie.device(xcve2302) {
    %shim = aie.tile(0, 0)
    %tile12 = aie.tile(1, 2)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @b(%tile12) Core {channelIndex = 0 : i32}
    // expected-error@+1 {{'aie.route' op has a packet header with no pkt_id; run --aie-assign-packet-ids before this pass}}
    aie.route from @a to [@b] {packet = #aie.packet_info<>}
  }
}

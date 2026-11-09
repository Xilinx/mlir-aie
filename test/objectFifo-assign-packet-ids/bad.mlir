//===- bad.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-packet-ids %s

// A pinned id has to fit the target's header field.

module {
  aie.device(xcve2302) {
    %shim = aie.tile(0, 0)
    %tile12 = aie.tile(1, 2)
    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @b(%tile12) Core
    // expected-error@+1 {{'aie.route' op pkt_id 32 is out of range (max 31)}}
    aie.route from @a to [@b] {packet = #aie.packet_info<pkt_id = 32>}
  }
}

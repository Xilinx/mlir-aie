//===- route_sources_bad.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A circuit joins one slave port to its master ports, so it cannot merge.

module {
  aie.device(xcve2302) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %tile12 = aie.tile(1, 2)
    aie.route_endpoint @a(%shim0) DMA
    aie.route_endpoint @b(%shim1) DMA
    aie.route_endpoint @c(%tile12) Core {channelIndex = 0 : i32}
    // expected-error@+1 {{'aie.route' op has several sources, so it needs a `packet` header; a circuit cannot merge streams}}
    aie.route from [@a, @b] to [@c]
  }
}

// -----

// The route is verified before the endpoints it names, so its own rule is
// what reports a name used twice.

module {
  aie.device(xcve2302) {
    %shim0 = aie.tile(0, 0)
    %tile12 = aie.tile(1, 2)
    // expected-error@+1 {{'aie.route' op names '@a' as a source twice}}
    aie.route from [@a, @a] to [@c] {packet = #aie.packet_info<>}
    aie.route_endpoint @a(%shim0) DMA
    aie.route_endpoint @c(%tile12) Core {channelIndex = 0 : i32}
  }
}

// -----

module {
  aie.device(xcve2302) {
    %shim0 = aie.tile(0, 0)
    %tile12 = aie.tile(1, 2)
    // expected-error@+1 {{'aie.route' op names '@a' more than once; an end is a source or a destination, and only one of each}}
    aie.route from @a to [@a, @c]
    aie.route_endpoint @a(%shim0) DMA
    aie.route_endpoint @c(%tile12) Core {channelIndex = 0 : i32}
  }
}

// -----

module {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    aie.route_endpoint @c(%tile12) Core {channelIndex = 0 : i32}
    // expected-error@+1 {{'aie.route' op expects at least one source}}
    aie.route from [] to [@c]
  }
}

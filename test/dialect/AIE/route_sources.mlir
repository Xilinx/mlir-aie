//===- route_sources.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

// One source prints bare, several print bracketed, and both parse.

// CHECK: aie.route from @a to [@c]
module {
  aie.device(xcve2302) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %tile12 = aie.tile(1, 2)
    aie.route_endpoint @a(%shim0) DMA
    aie.route_endpoint @b(%shim1) DMA
    aie.route_endpoint @c(%tile12) Core {channelIndex = 0 : i32}
    aie.route from @a to [@c]
  }
}

// -----

// CHECK: aie.route from [@a, @b] to [@c] {packet = #aie.packet_info<>}
module {
  aie.device(xcve2302) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %tile12 = aie.tile(1, 2)
    aie.route_endpoint @a(%shim0) DMA
    aie.route_endpoint @b(%shim1) DMA
    aie.route_endpoint @c(%tile12) Core {channelIndex = 0 : i32}
    aie.route from [@a, @b] to [@c] {packet = #aie.packet_info<>}
  }
}

// -----

// A bracketed single source parses and prints bare.
// CHECK: aie.route from @a to [@c]
module {
  aie.device(xcve2302) {
    %shim0 = aie.tile(0, 0)
    %tile12 = aie.tile(1, 2)
    aie.route_endpoint @a(%shim0) DMA
    aie.route_endpoint @c(%tile12) Core {channelIndex = 0 : i32}
    aie.route from [@a] to [@c]
  }
}

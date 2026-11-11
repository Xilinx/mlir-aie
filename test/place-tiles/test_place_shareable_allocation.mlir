//===- test_place_shareable_allocation.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An end a shareable shim allocation names may take turns on its channel:
// @a, on a logical shim pinned to (0, 0), shares MM2S 0 with @shared while
// MM2S 1 is reserved outright. Both placers place it there and allocation
// gives @a channel 0.

// RUN: aie-opt --verify-diagnostics --aie-place-tiles --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=2' --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s

// CHECK: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}, MM2S, 0, <pkt_id =

module @shareable_alloc_named {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %ls = aie.logical_tile<ShimNOCTile>(0, 0)
    %t2 = aie.tile(0, 2)
    aie.shim_dma_allocation @shared(%shim, MM2S, 0) {aie.shareable_with = [@a]}
    aie.shim_dma_allocation @reserved(%shim, MM2S, 1)
    aie.route_endpoint @a(%ls) DMA
    aie.route_endpoint @a_out(%t2) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.runtime_sequence @run(%input : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%input[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
    }
  }
}

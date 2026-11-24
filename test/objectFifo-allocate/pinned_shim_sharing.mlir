// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Sharing with the allocation does not permit sharing with its pinned user.
module @pinned_sharer {
  // expected-remark @+1 {{could not find a spill-aware allocation}}
  aie.device(npu2) {
    // expected-error @+1 {{requires at least 3 MM2S channels, but capacity is 2}}
    %home = aie.tile(0, 0)
    %alias = aie.logical_tile<ShimNOCTile>(0, 0)
    %core = aie.tile(0, 2)
    aie.shim_dma_allocation @shared(%home, MM2S, 0) {aie.shareable_with = [@pinned, @pending]}
    // expected-note @+1 {{pre-existing aie.shim_dma_allocation reserves DMA channel 1}}
    aie.shim_dma_allocation @reserved(%home, MM2S, 1)
    // expected-note @+1 {{DMA endpoint @pinned; occupies channel 0}}
    aie.route_endpoint @pinned(%alias) DMA {channelIndex = 0 : i32}
    // expected-note @+1 {{DMA endpoint @pending}}
    aie.route_endpoint @pending(%home) DMA
    aie.route_endpoint @a(%core) Core {channelIndex = 0 : i32}
    aie.route_endpoint @b(%core) Core {channelIndex = 1 : i32}
    aie.route from @pinned to [@a] {packet = #aie.packet_info<pkt_id = 0>}
    aie.route from @pending to [@b] {packet = #aie.packet_info<pkt_id = 1>}
  }
}

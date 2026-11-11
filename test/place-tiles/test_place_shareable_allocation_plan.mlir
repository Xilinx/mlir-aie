//===- test_place_shareable_allocation_plan.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The placers share a shareable shim allocation's channel as allocation does.
// @pinned: a route end pins @shared's channel on (0, 0), so @shared takes no
// listed end there; @a and @c, in flight together, fit only on (1, 0), where
// @a takes @shared1's channel. @listed: @shared's list alone lets @a take its
// channel, whatever their lifetimes.

// RUN: aie-opt --split-input-file --aie-place-tiles --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=2' --aie-assign-packet-ids --aie-objectfifo-allocate %s | FileCheck %s
// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' %s -o /dev/null
// RUN: aie-opt --split-input-file --pass-pipeline='builtin.module(aie-resource-stages)' %s | FileCheck %s --check-prefix=STAGES

// CHECK-LABEL: module @pinned
// CHECK-DAG: aie.shim_dma_allocation @r__shim_alloc(%{{.*}}tile_0_0, MM2S, 0,
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}tile_1_0, MM2S, 0,
// CHECK-DAG: aie.shim_dma_allocation @c__shim_alloc(%{{.*}}tile_1_0, MM2S, 1)
// CHECK-LABEL: module @listed
// CHECK-DAG: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}tile_0_0, MM2S, 0,
// CHECK-DAG: aie.shim_dma_allocation @c__shim_alloc(%{{.*}}tile_0_0, MM2S, 1)

// STAGES-LABEL: module @pinned
// STAGES: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}tile_1_0, MM2S, 0,
// STAGES-LABEL: module @listed
// STAGES: aie.shim_dma_allocation @a__shim_alloc(%{{.*}}tile_0_0, MM2S, 0,

// The SA, seed 1, does not move a lone tile with no nets: it reports the
// over-full position instead of calling it fitting.
module @pinned {
  // expected-warning@+1 {{over capacity (penalty=20)}}
  aie.device(npu2_2col) {
    %home = aie.tile(0, 0)
    %home1 = aie.tile(1, 0)
    %ls = aie.logical_tile<ShimNOCTile>(?, ?)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    aie.shim_dma_allocation @shared(%home, MM2S, 0) {aie.shareable_with = [@r, @a, @c]}
    aie.shim_dma_allocation @shared1(%home1, MM2S, 0) {aie.shareable_with = [@a, @c]}
    aie.route_endpoint @r(%home) DMA {channelIndex = 0 : i32}
    aie.route_endpoint @a(%ls) DMA
    aie.route_endpoint @c(%ls) DMA
    aie.route_endpoint @r_out(%c2) Core {channelIndex = 0 : i32}
    aie.route_endpoint @a_out(%c3) Core {channelIndex = 0 : i32}
    aie.route_endpoint @c_out(%c4) Core {channelIndex = 0 : i32}
    aie.route from @r to [@r_out] {packet = #aie.packet_info<pkt_id = 0>}
    aie.route from @a to [@a_out]
    aie.route from @c to [@c_out]
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @c, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @r, id = 2 : i64} : memref<16xi32>
    }
  }
}

// -----

module @listed {
  aie.device(npu2_1col) {
    %home = aie.tile(0, 0)
    %ls = aie.logical_tile<ShimNOCTile>(?, ?)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    aie.shim_dma_allocation @shared(%home, MM2S, 0) {aie.shareable_with = [@a]}
    aie.route_endpoint @a(%ls) DMA
    aie.route_endpoint @c(%ls) DMA
    aie.route_endpoint @a_out(%c3) Core {channelIndex = 0 : i32}
    aie.route_endpoint @c_out(%c4) Core {channelIndex = 0 : i32}
    aie.route from @a to [@a_out]
    aie.route from @c to [@c_out]
    aie.runtime_sequence @run(%x : memref<16xi32>, %y : memref<16xi32>, %z : memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%z[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @shared, id = 2 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%x[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @a, id = 0 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%y[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @c, id = 1 : i64} : memref<16xi32>
    }
  }
}

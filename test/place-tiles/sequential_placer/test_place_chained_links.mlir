//===- test_place_chained_links.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s

// @mid is one link's output and the next link's input. The two links fit one
// memtile's channels (3 in, 2 out), so they share it.

// CHECK-LABEL: module @chained_links_share_a_memtile
// CHECK: aie.objectfifo @mid(%mem_tile_0_1, {%mem_tile_0_1}, 2 : i32)
// CHECK: aie.objectfifo @out(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32)
module @chained_links_share_a_memtile {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %join = aie.logical_tile<MemTile>(?, ?)
    %fwd = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @a (%c0, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%c1, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%join, {%fwd}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%fwd, {%shim}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@mid] ([0, 16] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}

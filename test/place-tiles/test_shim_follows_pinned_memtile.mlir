//===- test_shim_follows_pinned_memtile.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An unpinned shim feeding a MemTile that is pinned to a column, where that
// same MemTile also carries unrelated traffic to cores in far-away columns.
// The shim's destination is the memtile, and the memtile's column is known,
// so the shim belongs in that column.
//
// The centroid used to resolve only CoreTile peers, so a memtile peer with a
// known column was discarded and the search fell through to the memtile's
// downstream cores. That returns the union of every column the memtile
// touches -- here {0, 6, 7} -- placing the shim near their mean instead of
// next to the memtile it actually feeds.
//
// Reduced from the weight feed of the fused LLM decode designs in mlir-air,
// where a shim feeding a col-1 memtile was placed on col 3.

// RUN: aie-opt --aie-place-tiles %s | FileCheck %s

// CHECK-LABEL: aie.device(npu2)
// The shim follows the memtile it feeds, not the memtile's other consumers.
// CHECK-DAG: aie.tile(1, 0)
// CHECK-NOT: aie.tile(3, 0)
// CHECK-NOT: aie.tile(4, 0)

module {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(1, ?)
    %core0 = aie.tile(0, 2)
    %core6 = aie.tile(6, 2)
    %core7 = aie.tile(7, 2)

    // The flow under test: host -> the pinned memtile.
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)

    // The memtile's other traffic, spread across the array. These are what
    // the core-only resolution used to collect for the shim.
    aie.flow(%mem, DMA : 0, %core0, DMA : 0)
    aie.flow(%mem, DMA : 1, %core6, DMA : 0)
    aie.flow(%mem, DMA : 2, %core7, DMA : 0)
  }
}

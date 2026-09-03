//===- test_shim_follows_pinned_memtile.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An unpinned shim feeding a col-pinned MemTile that also carries unrelated
// traffic to distant cores. Resolving past the memtile to those cores collects
// {0, 6, 7} and places the shim near their mean; the memtile's own column is
// the answer. Reduced from an mlir-air LLM decode weight feed.

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

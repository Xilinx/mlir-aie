//===- duplicate_core_ok.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Companion to duplicate_core.mlir: designs the DeviceOp::verify duplicate-core
// check must NOT reject. Cores on distinct tiles are fine; multiple non-core
// logical tiles at the same coordinate are fine (the placer merges them); cores
// on unplaced logical tiles defer the check until placement.

// RUN: aie-opt -split-input-file %s | FileCheck %s

// Distinct physical core tiles.
// CHECK-LABEL: @distinct_cores
module @distinct_cores {
  aie.device(npu2) {
    %t1 = aie.tile(0, 2)
    %t2 = aie.tile(0, 3)
    aie.core(%t1) { aie.end }
    aie.core(%t2) { aie.end }
  }
}

// -----

// Cores on unplaced logical tiles: no coordinates yet, so no collision.
// CHECK-LABEL: @unplaced_cores
module @unplaced_cores {
  aie.device(npu2) {
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
  }
}

// -----

// Two non-core logical tiles at the same coordinate is legal: the placer merges
// them onto one physical tile. Only cores are restricted to one-per-tile.
// CHECK-LABEL: @shared_memtile
module @shared_memtile {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %m1   = aie.logical_tile<MemTile>(1, 1)
    %m2   = aie.logical_tile<MemTile>(1, 1)
    %c1   = aie.logical_tile<CoreTile>(0, 2)
    %c2   = aie.logical_tile<CoreTile>(0, 3)
    aie.objectfifo @f1(%shim, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%m1,   {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%m2,   {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
  }
}

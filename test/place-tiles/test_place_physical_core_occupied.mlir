//===- test_place_physical_core_occupied.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=3" %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles %s | FileCheck %s
// RUN: aie-opt --aie-place-tiles="cores-per-col=4" %s | FileCheck %s

// The physical tile (0, 3) already runs its own aie.core, so neither placer
// offers it to a logical core. Before, both could put one there and the
// pass failed ("tile (0, 3) already has a core").
// CHECK: aie.core(%tile_0_3)
// CHECK-NOT: aie.core(%tile_0_3)

module @physical_core_occupied {
  aie.device(npu2_1col) {
    %pc = aie.tile(0, 3)
    %lc = aie.logical_tile<CoreTile>(?, ?)
    %lc2 = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%pc) {sym_name = "big"} : memref<16xi32>
    aie.objectfifo @f(%lc2, {%lc}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.core(%pc) { aie.end }
    aie.core(%lc) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<2048xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
    aie.core(%lc2) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<2048xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
  }
}

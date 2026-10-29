//===- test_place_chained_links.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles --split-input-file %s | FileCheck %s

// @mid is one link's output and the next link's input. The two links fit one
// memtile's channels (3 in, 2 out), but sharing it would loop @mid back into
// the tile it leaves, so they land on two.

// CHECK-LABEL: module @chained_links_on_two_memtiles
// CHECK: aie.objectfifo @mid(%mem_tile_0_1, {%mem_tile_1_1}, 2 : i32)
// CHECK: aie.objectfifo @out(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32)
module @chained_links_on_two_memtiles {
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

// -----

// CHECK-LABEL: module @chain_from_physical_memtile
// CHECK-DAG: %[[FROM:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[INTO:.*]] = aie.tile(1, 1)
// CHECK: aie.objectfifo @mid(%[[FROM]], {%[[INTO]]}, 2 : i32)
module @chain_from_physical_memtile {
  aie.device(npu2) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %join = aie.tile(0, 1)
    %fwd = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 3)
    aie.objectfifo @a (%c0, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%c1, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%join, {%fwd}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%fwd, {%shim}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@a, @b] -> [@mid] ([0, 16] [])
    aie.objectfifo.link [@mid] -> [@out] ([] [])
  }
}

// -----

// CHECK-LABEL: module @chain_into_physical_memtile
// CHECK-DAG: %[[FROM:.*]] = aie.tile(1, 1)
// CHECK-DAG: %[[INTO:.*]] = aie.tile(0, 1)
// CHECK: aie.objectfifo @mid(%[[FROM]], {%[[INTO]]}, 2 : i32)
module @chain_into_physical_memtile {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %join = aie.logical_tile<MemTile>(?, ?)
    %fwd = aie.tile(0, 1)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(0, 3)
    aie.objectfifo @a (%c0, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%c1, {%join}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @mid (%join, {%fwd}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out (%fwd, {%shim}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@mid] -> [@out] ([] [])
    aie.objectfifo.link [@a, @b] -> [@mid] ([0, 16] [])
  }
}

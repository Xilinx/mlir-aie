//===- test_place_linked_source_consumers.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A fifo that feeds a link can have other consumers besides the link tile:
// here a shim drains it as well. Only the link tile's side is the link's to
// count; the shim's input channel still has to fit the shim it lands on.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// @a feeds the memtile link and is drained by %shim_b. %shim_a's two drains
// take both input channels of the shim in column 0, so %shim_b goes to the
// next column, whichever order @a lists its consumers in.

// CHECK-LABEL: @link_tile_listed_first
// CHECK-DAG:     aie.tile(0, 0)
// CHECK-DAG:     aie.tile(1, 0)
module @link_tile_listed_first {
  aie.device(npu1) {
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(0, 3)
    %core_c = aie.tile(0, 4)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @f1(%core_a, {%shim_a}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @f2(%core_b, {%shim_a}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @a(%core_c, {%mem, %shim_b}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @b(%mem, {%core_a}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo.link [@a] -> [@b]([] [])
  }
}

// -----

// CHECK-LABEL: @shim_listed_first
// CHECK-DAG:     aie.tile(0, 0)
// CHECK-DAG:     aie.tile(1, 0)
module @shim_listed_first {
  aie.device(npu1) {
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(0, 3)
    %core_c = aie.tile(0, 4)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @f1(%core_a, {%shim_a}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @f2(%core_b, {%shim_a}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @a(%core_c, {%shim_b, %mem}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @b(%mem, {%core_a}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo.link [@a] -> [@b]([] [])
  }
}

//===- test_place_merge_memtile_streams.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two memtile LTOs, each fed by three cores in column 0. Together they fit one
// memtile's 6 S2MM channels, but only 4 streams reach a memtile from the core
// above it, and a memtile has no East/West ports, so the other 2 would have to
// detour through the shim row. The placer gives @mem_b the next memtile.
// With a single column there is no other memtile, and the merge goes ahead.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// CHECK-LABEL: @split_memtiles
// CHECK-DAG:   %[[M0:.*]] = aie.tile(0, 1)
// CHECK-DAG:   %[[M1:.*]] = aie.tile(1, 1)
// CHECK:       aie.objectfifo @a2({{.*}}, {%[[M0]]}
// CHECK:       aie.objectfifo @a3({{.*}}, {%[[M0]]}
// CHECK:       aie.objectfifo @a4({{.*}}, {%[[M0]]}
// CHECK:       aie.objectfifo @b2({{.*}}, {%[[M1]]}
// CHECK:       aie.objectfifo @b3({{.*}}, {%[[M1]]}
// CHECK:       aie.objectfifo @b4({{.*}}, {%[[M1]]}
module @split_memtiles {
  aie.device(npu2) {
    %c2 = aie.logical_tile<CoreTile>(0, 2)
    %c3 = aie.logical_tile<CoreTile>(0, 3)
    %c4 = aie.logical_tile<CoreTile>(0, 4)
    %mem_a = aie.logical_tile<MemTile>(?, ?)
    %mem_b = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a2(%c2, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a3(%c3, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a4(%c4, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @b2(%c2, {%mem_b}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @b3(%c3, {%mem_b}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @b4(%c4, {%mem_b}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
  }
}

// -----

// CHECK-LABEL: @one_column
// CHECK:       %[[M:.*]] = aie.tile(0, 1)
// CHECK:       aie.objectfifo @a2({{.*}}, {%[[M]]}
// CHECK:       aie.objectfifo @b2({{.*}}, {%[[M]]}
module @one_column {
  aie.device(npu2_1col) {
    %c2 = aie.logical_tile<CoreTile>(0, 2)
    %c3 = aie.logical_tile<CoreTile>(0, 3)
    %c4 = aie.logical_tile<CoreTile>(0, 4)
    %mem_a = aie.logical_tile<MemTile>(?, ?)
    %mem_b = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a2(%c2, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a3(%c3, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @a4(%c4, {%mem_a}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @b2(%c2, {%mem_b}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @b3(%c3, {%mem_b}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @b4(%c4, {%mem_b}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
  }
}

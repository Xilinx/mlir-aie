//===- test_place_routing_load.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s --implicit-check-not='aie.logical_tile'
// RUN: sed 's/%k = aie.logical_tile<MemTile>(?, ?)/%k = aie.tile(3, 1)/' %s | aie-opt --split-input-file --aie-place-tiles | FileCheck %s --implicit-check-not='aie.logical_tile'
// RUN: sed -E 's/(%%shared([2-4])) = aie.logical_tile<CoreTile>\(\?, \?\)/\1 = aie.logical_tile<CoreTile>(0, \2)/' %s | aie-opt --split-input-file --aie-place-tiles | FileCheck %s --implicit-check-not='aie.logical_tile'
// RUN: sed -E 's/(%%shared([2-4])) = aie.logical_tile<CoreTile>\(\?, \?\)/\1 = aie.tile(0, \2)/' %s | aie-opt --split-input-file --aie-place-tiles | FileCheck %s --implicit-check-not='aie.logical_tile'
// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform='skip-verify=true' --aie-create-pathfinder-flows %s -o /dev/null

// Two eight-way broadcasts have the same centroid. Spare DMA channels alone
// must not make them share a memtile; use the adjacent column instead.
// CHECK-LABEL: @broadcast_load
// CHECK-DAG: aie.tile(3, 1)
// CHECK-DAG: aie.tile({{[24]}}, 1)
module @broadcast_load {
  aie.device(npu2) {
    %k = aie.logical_tile<MemTile>(?, ?)
    %v = aie.logical_tile<MemTile>(?, ?)
    %c0 = aie.tile(0, 2)
    %c1 = aie.tile(1, 2)
    %c2 = aie.tile(2, 2)
    %c3 = aie.tile(3, 2)
    %c4 = aie.tile(4, 2)
    %c5 = aie.tile(5, 2)
    %c6 = aie.tile(6, 2)
    %c7 = aie.tile(7, 2)
    aie.objectfifo @k(%k, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @v(%v, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Each core needs two independent streams from below. Three cores fit the
// column's six incoming streams; the fourth should start another column.
// CHECK-LABEL: @column_input_load
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(0, 3)
// CHECK-DAG: aie.tile(0, 4)
// CHECK-DAG: aie.tile(1, 2)
// CHECK-NOT: aie.tile(0, 5)
module @column_input_load {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(?, ?)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(?, ?)
    %m1 = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a0(%m0, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%m1, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a1(%m0, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%m1, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a2(%m0, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%m1, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a3(%m0, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b3(%m1, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Four consumers of the same two broadcasts need only two incoming streams,
// including when the first three consumers are pinned or physical tiles.
// Fix the last row so it cannot merge into an existing physical consumer.
// CHECK-LABEL: @shared_column_inputs
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(0, 3)
// CHECK-DAG: aie.tile(0, 4)
// CHECK-DAG: aie.tile(0, 5)
module @shared_column_inputs {
  aie.device(npu2) {
    %shared2 = aie.logical_tile<CoreTile>(?, ?)
    %shared3 = aie.logical_tile<CoreTile>(?, ?)
    %shared4 = aie.logical_tile<CoreTile>(?, ?)
    %shared5 = aie.logical_tile<CoreTile>(?, 5)
    %m = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a(%m, {%shared2, %shared3, %shared4, %shared5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b(%m, {%shared2, %shared3, %shared4, %shared5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// The routing budget is soft: column hints can still use all four cores even
// when their eight distinct incoming streams exceed the column budget.
// CHECK-LABEL: @column_load_fallback
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(0, 3)
// CHECK-DAG: aie.tile(0, 4)
// CHECK-DAG: aie.tile(0, 5)
module @column_load_fallback {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, ?)
    %c1 = aie.logical_tile<CoreTile>(0, ?)
    %c2 = aie.logical_tile<CoreTile>(0, ?)
    %c3 = aie.logical_tile<CoreTile>(0, ?)
    %m0 = aie.logical_tile<MemTile>(?, ?)
    %m1 = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a0(%m0, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%m1, {%c0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a1(%m0, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%m1, {%c1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a2(%m0, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%m1, {%c2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a3(%m0, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b3(%m1, {%c3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Pinned cores contribute to outgoing column pressure too. Four streams out
// already fill column 0, even though its two remaining compute tiles are free.
// CHECK-LABEL: @column_output_load
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(0, 3)
// CHECK-DAG: aie.tile(1, 2)
module @column_output_load {
  aie.device(npu2) {
    %c0 = aie.logical_tile<CoreTile>(0, 2)
    %c1 = aie.logical_tile<CoreTile>(0, 3)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(?, ?)
    %m1 = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a0(%c0, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b0(%c0, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a1(%c1, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b1(%c1, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @a2(%c2, {%m0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b2(%c2, {%m1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

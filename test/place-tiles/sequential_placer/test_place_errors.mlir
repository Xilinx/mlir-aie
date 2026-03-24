//===- test_place_errors.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles %s 2>&1 | FileCheck %s

module @three_inputs_exceeds_capacity {
  aie.device(npu1) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)

    // CHECK: error: tile requires 3 input/0 output DMA channels, but only 2 input/2 output available
    %core = aie.logical_tile<CoreTile>(?, ?)

    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in3 (%shim3, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
  }
}

// -----

module @three_outputs_exceeds_capacity {
  aie.device(npu1) {
    // CHECK: error: tile (0, 2) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %core = aie.logical_tile<CoreTile>(0, 2)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)

    aie.objectfifo @out1 (%core, {%mem1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2 (%core, {%mem2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out3 (%core, {%mem3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
  }
}

// -----

// MemTile DMA exhaustion on single-column device
module @memtile_exhaustion {
  aie.device(npu1_1col) {
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)
    %core4 = aie.logical_tile<CoreTile>(?, ?)

    // First 6 memtiles merge, using all 6 output channels
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)
    %mem4 = aie.logical_tile<MemTile>(?, ?)
    %mem5 = aie.logical_tile<MemTile>(?, ?)
    %mem6 = aie.logical_tile<MemTile>(?, ?)
    // CHECK: error: no MemTile with sufficient DMA capacity
    %mem7 = aie.logical_tile<MemTile>(?, ?)

    aie.objectfifo @of1 (%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%mem2, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of3 (%mem3, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of4 (%mem4, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of5 (%mem5, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of6 (%mem6, {%core4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @of7 (%mem7, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
    aie.core(%core4) { aie.end }
  }
}

// -----

// ShimPLTile has no DMA, unconstrained placement not supported
module @shimpl_unsupported {
  aie.device(xcvc1902) {
    // CHECK: error: DMA channel-based SequentialPlacer does not support unplaced ShimPLTiles
    %shim = aie.logical_tile<ShimPLTile>(?, ?)
  }
}

// -----

// Column constraint cannot be satisfied (all tiles in column taken)
module @col_constraint_exhausted {
  aie.device(npu1_1col) {
    // npu1_1col has only column 0 with 4 core rows (2-5)
    %c1 = aie.logical_tile<CoreTile>(0, 2)
    %c2 = aie.logical_tile<CoreTile>(0, 3)
    %c3 = aie.logical_tile<CoreTile>(0, 4)
    %c4 = aie.logical_tile<CoreTile>(0, 5)
    // CHECK: error: no compute tile available matching constraint (0, ?)
    %c5 = aie.logical_tile<CoreTile>(0, ?)

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.core(%c4) { aie.end }
    aie.core(%c5) { aie.end }
  }
}

// -----

// Row constraint cannot be satisfied (all tiles in row taken)
module @row_constraint_exhausted {
  aie.device(npu1) {
    // npu1 has 4 columns (0-3), take all row 2 tiles
    %c1 = aie.logical_tile<CoreTile>(0, 2)
    %c2 = aie.logical_tile<CoreTile>(1, 2)
    %c3 = aie.logical_tile<CoreTile>(2, 2)
    %c4 = aie.logical_tile<CoreTile>(3, 2)
    // CHECK: error: no compute tile available matching constraint (?, 2)
    %c5 = aie.logical_tile<CoreTile>(?, 2)

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.core(%c4) { aie.end }
    aie.core(%c5) { aie.end }
  }
}

// -----

// ShimNOCTile DMA exhaustion on single-column device
module @shimnoc_exhaustion {
  aie.device(npu1_1col) {
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)

    // First two shims merge, using 2 output channels
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK: error: no ShimNOCTile with sufficient DMA capacity
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)

    aie.objectfifo @of1 (%shim1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%shim2, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @of3 (%shim3, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
  }
}

// -----

// All compute tiles exhausted (unconstrained)
module @compute_tiles_exhausted {
  aie.device(npu1_1col) {
    // npu1_1col has 4 core tiles (rows 2-5 in column 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK: error: no available compute tiles for placement
    %c5 = aie.logical_tile<CoreTile>(?, ?)

    aie.core(%c1) { aie.end }
    aie.core(%c2) { aie.end }
    aie.core(%c3) { aie.end }
    aie.core(%c4) { aie.end }
    aie.core(%c5) { aie.end }
  }
}

// -----

// Mixed input/output exceeds capacity
module @mixed_channels_exceed_capacity {
  aie.device(npu1) {
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)

    // CHECK: error: tile requires 2 input/3 output DMA channels, but only 2 input/2 output available
    %core = aie.logical_tile<CoreTile>(?, ?)

    // 2 inputs
    aie.objectfifo @in1 (%shim1, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2 (%shim2, {%core}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    // 3 outputs - exceeds capacity
    aie.objectfifo @out1 (%core, {%mem1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2 (%core, {%mem2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out3 (%core, {%mem3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core) { aie.end }
  }
}

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

    // First MemTile uses all 6 output channels
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    // CHECK: error: no MemTile with sufficient DMA capacity
    %mem2 = aie.logical_tile<MemTile>(?, ?)

    aie.objectfifo @of1 (%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of3 (%mem1, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of4 (%mem1, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of5 (%mem1, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of6 (%mem1, {%core4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @of7 (%mem2, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

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

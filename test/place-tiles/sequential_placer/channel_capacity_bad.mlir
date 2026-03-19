//===- channel_capacity_bad.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles %s 2>&1 | FileCheck %s

// This test verifies that the sequential placer correctly validates DMA channel
// capacity for ObjectFifos between cores and non-core tiles (shim/mem).

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

// Test: MemTile DMA exhaustion with single-column device
// npu1_1col has only 1 MemTile at (0,1) with 6 DMA channels total
// We exhaust the MemTile's output channels so the second logical MemTile
// cannot be placed, triggering "no MemTile with sufficient DMA capacity"
module @memtile_dma_exhaustion {
  aie.device(npu1_1col) {
    // 4 cores, each receiving 1-2 inputs (within core's 2-input limit)
    %core1 = aie.logical_tile<CoreTile>(?, ?)
    %core2 = aie.logical_tile<CoreTile>(?, ?)
    %core3 = aie.logical_tile<CoreTile>(?, ?)
    %core4 = aie.logical_tile<CoreTile>(?, ?)

    // First logical MemTile - uses all 6 output channels of physical (0,1)
    %mem1 = aie.logical_tile<MemTile>(?, ?)

    // Second logical MemTile - no capacity left on (0,1), and no other MemTile exists
    // CHECK: error: no MemTile with sufficient DMA capacity
    %mem2 = aie.logical_tile<MemTile>(?, ?)

    // mem1 outputs to 4 different cores (6 total outputs, exhausts MemTile)
    aie.objectfifo @of1 (%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%mem1, {%core1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of3 (%mem1, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of4 (%mem1, {%core2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of5 (%mem1, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of6 (%mem1, {%core4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // mem2 needs at least 1 output channel, but MemTile (0,1) is exhausted
    aie.objectfifo @of7 (%mem2, {%core3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.core(%core1) { aie.end }
    aie.core(%core2) { aie.end }
    aie.core(%core3) { aie.end }
    aie.core(%core4) { aie.end }
  }
}

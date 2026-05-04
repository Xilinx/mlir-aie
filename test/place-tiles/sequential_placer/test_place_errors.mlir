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

// -----

// Pinned CoreTile buffer exceeds L1 (npu1: 64KB).
module @buffer_overflow_pinned {
  aie.device(npu1) {
    // CHECK: error: tile (2, 3) requires 132096 bytes for buffers + stack, but only 65536 bytes available
    %t = aie.logical_tile<CoreTile>(2, 3)
    %b = aie.buffer(%t) : memref<32768xi32>
    aie.core(%t) { aie.end }
  }
}

// -----

// Unconstrained CoreTile, buffer exceeds L1 on every candidate.
module @buffer_overflow_unconstrained {
  aie.device(npu1) {
    // CHECK: error: no available compute tiles for placement (buffer capacity exceeded on every candidate)
    %t = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%t) : memref<32768xi32>
    aie.core(%t) { aie.end }
  }
}

// -----

// Per-LogicalTileOp summation: 2 * 32KB + 1KB stack > 64KB.
module @buffer_overflow_sum {
  aie.device(npu1) {
    // CHECK: error: tile (2, 3) requires 66560 bytes for buffers + stack, but only 65536 bytes available
    %t = aie.logical_tile<CoreTile>(2, 3)
    %b0 = aie.buffer(%t) : memref<8192xi32>
    %b1 = aie.buffer(%t) : memref<8192xi32>
    aie.core(%t) { aie.end }
  }
}

// -----

// MemTile per-LogicalTileOp upper bound (npu1: 512KB).
module @memtile_buffer_overflow {
  aie.device(npu1) {
    // CHECK: error: tile (0, 1) requires 800000 bytes for buffers, but only 524288 bytes available
    %mt = aie.logical_tile<MemTile>(0, 1)
    %b = aie.buffer(%mt) : memref<200000xi32>
  }
}

// -----

// AIE2P (npu2) target-model dispatch.
module @buffer_overflow_npu2 {
  aie.device(npu2) {
    // CHECK: error: tile (2, 3) requires 132096 bytes for buffers + stack, but only 65536 bytes available
    %t = aie.logical_tile<CoreTile>(2, 3)
    %b = aie.buffer(%t) : memref<32768xi32>
    aie.core(%t) { aie.end }
  }
}

// -----

// Stack-only demand: pathological CoreOp stack size alone exceeds L1.
module @buffer_overflow_stack_only {
  aie.device(npu1) {
    // CHECK: error: tile (2, 3) requires 131072 bytes for buffers + stack, but only 65536 bytes available
    %t = aie.logical_tile<CoreTile>(2, 3)
    aie.core(%t) { aie.end } { stack_size = 131072 : i32 }
  }
}

// -----

// Two LogicalTileOps, second one overflows. Confirms each LogicalTileOp's
// budget is checked independently and the placer reports the offending tile,
// not the fitting one.
module @buffer_overflow_second_of_two {
  aie.device(npu1) {
    // CHECK: error: tile (2, 4) requires 132096 bytes for buffers + stack, but only 65536 bytes available
    %t1 = aie.logical_tile<CoreTile>(2, 3)
    %b1 = aie.buffer(%t1) : memref<1024xi32>
    aie.core(%t1) { aie.end }
    %t2 = aie.logical_tile<CoreTile>(2, 4)
    %b2 = aie.buffer(%t2) : memref<32768xi32>
    aie.core(%t2) { aie.end }
  }
}

// -----

// Two LogicalTileOps both pinned at (2, 3): individual budgets fit
// (33KB each < 64KB), but the placer collapses them onto one physical
// CoreTile so the L1 demand sums to ~67KB. The per-physical-tile
// accumulator must catch it; without it, AIEAssignBuffers would report
// the overflow downstream against a tile the user never wrote.
module @buffer_overflow_two_pinned_same_tile {
  aie.device(npu1) {
    // CHECK: error: tile (2, 3) requires 33792 bytes for buffers + stack (plus 33792 bytes already charged from earlier placements), but only 65536 bytes available
    %t1 = aie.logical_tile<CoreTile>(2, 3)
    %b1 = aie.buffer(%t1) : memref<8192xi32>
    aie.core(%t1) { aie.end }
    %t2 = aie.logical_tile<CoreTile>(2, 3)
    %b2 = aie.buffer(%t2) : memref<8192xi32>
    aie.core(%t2) { aie.end }
  }
}

// -----

// Same accumulation for MemTile sharing across LogicalTileOps. The
// existing placer permits multiple MemTile LogicalTileOps to land on
// one physical MemTile (cf. `removeTile` not removing MemTile in the
// constrained branch); summed buffer bytes must respect MemTile capacity.
module @buffer_overflow_two_pinned_same_memtile {
  aie.device(npu1) {
    // CHECK: error: tile (0, 1) requires 400000 bytes for buffers (plus 400000 bytes already charged from earlier placements), but only 524288 bytes available
    %m1 = aie.logical_tile<MemTile>(0, 1)
    %b1 = aie.buffer(%m1) : memref<100000xi32>
    %m2 = aie.logical_tile<MemTile>(0, 1)
    %b2 = aie.buffer(%m2) : memref<100000xi32>
  }
}

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

// Flow-derived MM2S demand exceeds shim capacity.
module @flow_shim_output_exceeds_capacity {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    aie.flow(%shim, DMA : 0, %c1, DMA : 0)
    aie.flow(%shim, DMA : 1, %c2, DMA : 0)
    aie.flow(%shim, DMA : 2, %c3, DMA : 0)
  }
}

// -----

// Distinct source DMA channels are NOT deduplicated: three flows on channels
// 0, 1, 2 still consume three MM2S budgets (only same-channel broadcasts
// dedup). This guards against an over-eager dedup that would mask real
// capacity overflows.
module @flow_distinct_channels_no_dedup {
  aie.device(npu1) {
    // CHECK: error: tile (0, 0) requires 0 input/3 output DMA channels, but only 2 input/2 output available
    %shim = aie.logical_tile<ShimNOCTile>(0, 0)
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    aie.flow(%shim, DMA : 0, %c1, DMA : 0)
    aie.flow(%shim, DMA : 1, %c2, DMA : 0)
    aie.flow(%shim, DMA : 2, %c3, DMA : 0)
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

// Both endpoints pinned far apart: not memory-affinity neighbors.
module @buffer_adjacency_both_pinned_violation {
  aie.device(npu1) {
    // CHECK: error: tile (0, 2) violates shared-L1 buffer adjacency
    // CHECK: note: shared-L1 buffer consumer peer placed at (3, 5)
    %owner = aie.logical_tile<CoreTile>(0, 2)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(3, 5)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Pinned owner + unconstrained consumer with a column constraint that has no
// memory-affinity slot relative to the owner.
module @buffer_adjacency_unsatisfiable_column {
  aie.device(npu1) {
    // CHECK: error: no compute tile available matching constraint (3, ?) and shared-L1 buffer adjacency
    // CHECK: note: shared-L1 buffer owner peer placed at (0, 2)
    %owner = aie.logical_tile<CoreTile>(0, 2)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %consumer = aie.logical_tile<CoreTile>(3, ?)
    aie.core(%consumer) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

// -----

// Star with too many consumers: an owner can host at most 3 cross-tile
// consumers (W, N, S — E is internal in AIE2). A 4th unconstrained consumer
// has no affinity slot left.
module @buffer_adjacency_star_oversubscribed {
  aie.device(npu1) {
    %owner = aie.logical_tile<CoreTile>(1, 3)
    %buf   = aie.buffer(%owner) : memref<16xi32>
    aie.core(%owner) { aie.end }
    %c1 = aie.logical_tile<CoreTile>(?, ?)
    %c2 = aie.logical_tile<CoreTile>(?, ?)
    %c3 = aie.logical_tile<CoreTile>(?, ?)
    // CHECK: error: no available compute tiles for placement (shared-L1 buffer adjacency unsatisfiable)
    // CHECK: note: shared-L1 buffer owner peer placed at (1, 3)
    %c4 = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%c1) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c2) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c3) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
    aie.core(%c4) {
      %i = arith.constant 0 : index
      %v = memref.load %buf[%i] : memref<16xi32>
      aie.end
    }
  }
}

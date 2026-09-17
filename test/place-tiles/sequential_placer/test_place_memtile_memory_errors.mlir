//===- test_place_memtile_memory_errors.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles %s 2>&1 | FileCheck %s

// DMA capacity alone would allow this merge, but its buffers do not fit.
// CHECK: error: no MemTile has sufficient local memory for 300000 bytes with the required DMA capacity (local memory capacity 524288 bytes per tile)
// CHECK: note: automatic MemTile placement keeps buffers local because spilling restricts DMA channel availability
module @exhausted {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %a = aie.logical_tile<MemTile>(?, ?)
    %b = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a(%shim, {%a}, 2 : i32) : !aie.objectfifo<memref<150000xi8>>
    aie.objectfifo @b(%shim, {%b}, 2 : i32) : !aie.objectfifo<memref<150000xi8>>
  }
}

// -----

// Fully constrained logical tiles must obey the same memory budget.
// CHECK: error: MemTile (0, 1) requires 600000 bytes of local memory, but capacity is 524288
// CHECK: note: automatic MemTile placement keeps buffers local because spilling restricts DMA channel availability
module @pinned {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %a = aie.logical_tile<MemTile>(0, 1)
    %b = aie.logical_tile<MemTile>(0, 1)
    aie.objectfifo @a(%shim, {%a}, 2 : i32) : !aie.objectfifo<memref<150000xi8>>
    aie.objectfifo @b(%shim, {%b}, 2 : i32) : !aie.objectfifo<memref<150000xi8>>
  }
}

// -----

// A fixed tile's overflowing pools could otherwise spill into the memory
// reserved for an automatically placed logical tile.
// CHECK: error: automatic MemTile placement requires local buffers: 600000 bytes required, but capacity is 524288; spilling restricts DMA channel availability
module @fixed_neighbor {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %fixed = aie.tile(0, 1)
    %logical = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @fixed(%shim, {%fixed}, 2 : i32) : !aie.objectfifo<memref<300000xi8>>
    aie.objectfifo @logical(%shim, {%logical}, 2 : i32) : !aie.objectfifo<memref<16xi8>>
  }
}

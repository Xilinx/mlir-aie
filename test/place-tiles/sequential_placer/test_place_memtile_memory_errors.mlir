//===- test_place_memtile_memory_errors.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s -o /dev/null
// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform="skip-verify=true" --verify-diagnostics %s -o /dev/null

// A one-column device has no neighbor to spill to. Diagnose actual allocation,
// not a speculative memory estimate in placement. Placement success alone is
// not a memory-feasibility guarantee; allocation in the stateful transform
// diagnoses the exhausted MemTile.
module @exhausted {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %a = aie.logical_tile<MemTile>(?, ?)
    %b = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @a(%shim, {%a}, 2 : i32) : !aie.objectfifo<memref<150000xi8>>
    // expected-error @+1 {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo @b(%shim, {%b}, 2 : i32) : !aie.objectfifo<memref<150000xi8>>
  }
}

// -----

// Fully constrained logical tiles can still spill automatically.
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

// Mixing fixed and logical tiles must not disable spilling either.
module @fixed_neighbor {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %fixed = aie.tile(0, 1)
    %logical = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo @fixed(%shim, {%fixed}, 2 : i32) : !aie.objectfifo<memref<300000xi8>>
    aie.objectfifo @logical(%shim, {%logical}, 2 : i32) : !aie.objectfifo<memref<16xi8>>
  }
}

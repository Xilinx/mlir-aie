//===- test_place_spread_unanchored_tiles.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Exercises the `spread-unanchored-tiles` pass option on aie-place-tiles.
// computeCentroidColumn derives a non-core LTO's target column from its
// CoreTile peers (directly, or through an already-placed non-core peer since
// #3551); a design with no compute tile anywhere still has no anchor at all,
// so the LTOs that start a chain fall back to column 0 and only the DMA
// channel count on that column's tiles forces an eventual spill. With the
// option, an unanchored LTO is ranked by DMA load instead, in the direction
// it actually needs.
//
// --implicit-check-not bounds every column to the ones named by a CHECK-DAG:
// a plain CHECK-NOT after a CHECK-DAG group only covers input past that
// group's furthest match, so a tile printed BEFORE it (declaration order is
// reverse-numeric here, not source order) would go unchecked.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s --check-prefix=PILE --implicit-check-not='aie.tile('
// RUN: aie-opt --split-input-file --aie-place-tiles='spread-unanchored-tiles=false' %s | FileCheck %s --check-prefix=PILE --implicit-check-not='aie.tile('
// RUN: aie-opt --split-input-file --aie-place-tiles='spread-unanchored-tiles=true' %s | FileCheck %s --check-prefix=SPREAD --implicit-check-not='aie.tile('

// Four independent shim->memtile->shim passthrough chains, no compute tile
// anywhere. Piled, each column's tiles take two chains before the next
// column is touched (a ShimNOCTile/MemTile on npu1 has 2 DMA channels per
// direction, getDMACapacity), so 4 chains fill exactly 2 columns instead of
// running 4-way. Spread, chain i lands entirely on column i -- shim and
// memtile together, which is what ranking by direction-matched load buys: a
// summed load would push each chain's read and write onto different
// columns.

// PILE-LABEL:     @anchorless_passthrough
// PILE-DAG:       aie.tile(0, 0)
// PILE-DAG:       aie.tile(0, 1)
// PILE-DAG:       aie.tile(1, 0)
// PILE-DAG:       aie.tile(1, 1)

// SPREAD-LABEL:   @anchorless_passthrough
// SPREAD-DAG:     aie.tile(0, 0)
// SPREAD-DAG:     aie.tile(1, 0)
// SPREAD-DAG:     aie.tile(2, 0)
// SPREAD-DAG:     aie.tile(3, 0)
// SPREAD-DAG:     aie.tile(0, 1)
// SPREAD-DAG:     aie.tile(1, 1)
// SPREAD-DAG:     aie.tile(2, 1)
// SPREAD-DAG:     aie.tile(3, 1)
module @anchorless_passthrough {
  aie.device(npu1) {
    %shim0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem0 = aie.logical_tile<MemTile>(?, ?)
    %shim1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem1 = aie.logical_tile<MemTile>(?, ?)
    %shim2 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem2 = aie.logical_tile<MemTile>(?, ?)
    %shim3 = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem3 = aie.logical_tile<MemTile>(?, ?)
    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)
    aie.flow(%mem0, DMA : 0, %shim0, DMA : 0)
    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)
    aie.flow(%mem1, DMA : 0, %shim1, DMA : 0)
    aie.flow(%shim2, DMA : 0, %mem2, DMA : 0)
    aie.flow(%mem2, DMA : 0, %shim2, DMA : 0)
    aie.flow(%shim3, DMA : 0, %mem3, DMA : 0)
    aie.flow(%mem3, DMA : 0, %shim3, DMA : 0)
  }
}

// -----

// One CoreTile is enough to anchor both non-core LTOs, so the option changes
// nothing: the centroid is real and distance still leads.

// PILE-LABEL:     @anchored_is_unaffected
// PILE-DAG:       aie.tile(0, 0)
// PILE-DAG:       aie.tile(0, 1)
// PILE-DAG:       aie.tile(0, 2)

// SPREAD-LABEL:   @anchored_is_unaffected
// SPREAD-DAG:     aie.tile(0, 0)
// SPREAD-DAG:     aie.tile(0, 1)
// SPREAD-DAG:     aie.tile(0, 2)
module @anchored_is_unaffected {
  aie.device(npu1) {
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem = aie.logical_tile<MemTile>(?, ?)
    %core = aie.tile(0, 2)
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%mem, DMA : 0, %core, DMA : 0)
  }
}

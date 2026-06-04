//===- test_place_merge_logical_tiles.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Exercises the `merge-logical-tiles` pass option on aie-place-tiles. The
// option gates whether multiple non-core aie.logical_tile ops may share
// one physical aie.tile when DMA channel capacity permits. The default
// (true) keeps existing behavior. With false, findTileWithCapacity rejects
// any tile that already hosts a non-core LTO. The option only affects
// ShimNOCTile and MemTile placement; CoreTile placement is unchanged.

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s --check-prefix=MERGE
// RUN: aie-opt --split-input-file --aie-place-tiles='merge-logical-tiles=true' %s | FileCheck %s --check-prefix=MERGE
// RUN: aie-opt --split-input-file --aie-place-tiles='merge-logical-tiles=false' %s | FileCheck %s --check-prefix=NOMERGE

// Two ShimNOC LTOs each carrying one packet flow to a different core.
// The placer dedups channel requirements by (LTO, channel), so each LTO
// reports needing only 1 output channel; with merging on, both fit on one
// physical shim. With merging off, each gets its own.

// MERGE-LABEL:   @two_shim_packet_flows
// MERGE:         aie.tile(0, 0)
// MERGE-NOT:     aie.tile(1, 0)

// NOMERGE-LABEL: @two_shim_packet_flows
// NOMERGE-DAG:   aie.tile(0, 0)
// NOMERGE-DAG:   aie.tile(1, 0)
// NOMERGE-NOT:   aie.tile(2, 0)
module @two_shim_packet_flows {
  aie.device(npu1) {
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(1, 2)
    aie.packet_flow(0) {
      aie.packet_source<%shim_a, DMA : 0>
      aie.packet_dest<%core_a, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_b, DMA : 0>
      aie.packet_dest<%core_b, DMA : 0>
    }
  }
}

// -----

// Same idea for memtiles. Merging on -> one memtile col; off -> spread.

// MERGE-LABEL:   @two_memtile_flows
// MERGE:         aie.tile(0, 1)
// MERGE-NOT:     aie.tile(1, 1)

// NOMERGE-LABEL: @two_memtile_flows
// NOMERGE-DAG:   aie.tile(0, 1)
// NOMERGE-DAG:   aie.tile(1, 1)
// NOMERGE-NOT:   aie.tile(2, 1)
module @two_memtile_flows {
  aie.device(npu1) {
    %mem_a = aie.logical_tile<MemTile>(?, ?)
    %mem_b = aie.logical_tile<MemTile>(?, ?)
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(1, 2)
    aie.flow(%mem_a, DMA : 0, %core_a, DMA : 0)
    aie.flow(%mem_b, DMA : 0, %core_b, DMA : 0)
  }
}

// -----

// Mixing both LTO types: 2 shim + 2 memtile. Each pair behaves as above.

// MERGE-LABEL:   @mixed_shim_memtile
// MERGE-DAG:     aie.tile(0, 0)
// MERGE-DAG:     aie.tile(0, 1)
// MERGE-NOT:     aie.tile(1, 0)
// MERGE-NOT:     aie.tile(1, 1)

// NOMERGE-LABEL: @mixed_shim_memtile
// NOMERGE-DAG:   aie.tile(0, 0)
// NOMERGE-DAG:   aie.tile(1, 0)
// NOMERGE-DAG:   aie.tile(0, 1)
// NOMERGE-DAG:   aie.tile(1, 1)
module @mixed_shim_memtile {
  aie.device(npu1) {
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    %mem_a = aie.logical_tile<MemTile>(?, ?)
    %mem_b = aie.logical_tile<MemTile>(?, ?)
    %core_a = aie.tile(0, 2)
    %core_b = aie.tile(1, 2)
    aie.flow(%shim_a, DMA : 0, %mem_a, DMA : 0)
    aie.flow(%mem_a, DMA : 1, %core_a, DMA : 0)
    aie.flow(%shim_b, DMA : 0, %mem_b, DMA : 0)
    aie.flow(%mem_b, DMA : 1, %core_b, DMA : 0)
  }
}

// -----

// CoreTile placement is unaffected by merge-logical-tiles. Two unhinted
// cores land at distinct (col, row) under both settings (column-major).

// MERGE-LABEL:   @cores_unaffected
// MERGE-DAG:     aie.tile(0, 2)
// MERGE-DAG:     aie.tile(0, 3)

// NOMERGE-LABEL: @cores_unaffected
// NOMERGE-DAG:   aie.tile(0, 2)
// NOMERGE-DAG:   aie.tile(0, 3)
module @cores_unaffected {
  aie.device(npu1) {
    %core_a = aie.logical_tile<CoreTile>(?, ?)
    %core_b = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%core_a) { aie.end }
    aie.core(%core_b) { aie.end }
  }
}

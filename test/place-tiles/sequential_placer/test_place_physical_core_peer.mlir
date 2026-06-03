//===- test_place_physical_core_peer.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Regression: when the connected core is a physical aie.tile (not an LTO),
// computeCentroidColumn must still count its column. Pre-fix the BFS hit
// the TileOp peer, then early-returned without summing it, so the centroid
// stayed at 0 and the shim landed in column 0 regardless of which column
// the core lived in.

// CHECK-LABEL: @shim_centroid_from_physical_core
module @shim_centroid_from_physical_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(3, 2)
    %core = aie.tile(3, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(3, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    // CHECK: aie.flow(%[[SHIM]], DMA : 0, %[[CORE]], DMA : 0)
    aie.flow(%shim, DMA : 0, %core, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Same regression via packet_flow: physical core at col 5, shim should
// follow the centroid to col 5 instead of falling back to col 0.

// CHECK-LABEL: @shim_centroid_from_physical_core_packet
module @shim_centroid_from_physical_core_packet {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(2, 2)
    %core = aie.tile(2, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(2, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.packet_flow(0x1) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%core, DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Centroid averaged across multiple physical-core peers. Shim feeds a
// 2-core broadcast at cols 0 and 4; centroid rounds to col 2.

// CHECK-LABEL: @shim_centroid_averaging_physical_cores
module @shim_centroid_averaging_physical_cores {
  aie.device(npu2) {
    // CHECK-DAG: %[[C0:.*]] = aie.tile(0, 2)
    %c0 = aie.tile(0, 2)
    // CHECK-DAG: %[[C4:.*]] = aie.tile(4, 2)
    %c4 = aie.tile(4, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(2, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.packet_flow(0x2) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c0, DMA : 0>
      aie.packet_dest<%c4, DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Centroid propagates through an unplaced memtile LTO to the physical
// core: shim is two adjacencies away from the core but should still
// inherit the core's column.

// CHECK-LABEL: @shim_centroid_through_memtile_to_physical_core
module @shim_centroid_through_memtile_to_physical_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(3, 2)
    %core = aie.tile(3, 2)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(3, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(3, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%mem, DMA : 1, %core, DMA : 0)
    // CHECK-NOT: aie.logical_tile
  }
}

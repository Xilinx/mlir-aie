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

// Same regression via packet_flow: physical core at col 2, shim should
// follow the centroid to col 2 instead of falling back to col 0.

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

// -----

// Four single-dest flows at col 0 (3 packet + 1 circuit) plus an 8-way
// broadcast over cols 0..7. The broadcast's cost is flat inside its span,
// so the shim lands at col 0. Pre-fix mean-of-cores gave col 3.

// CHECK-LABEL: @shim_routing_cost_broadcast_vs_single_dest
module @shim_routing_cost_broadcast_vs_single_dest {
  aie.device(npu2) {
    // CHECK-DAG: %[[C04:.*]] = aie.tile(0, 4)
    %c04 = aie.tile(0, 4)
    // CHECK-DAG: %[[C03:.*]] = aie.tile(0, 3)
    %c03 = aie.tile(0, 3)
    // CHECK-DAG: %[[C02:.*]] = aie.tile(0, 2)
    %c02 = aie.tile(0, 2)
    // CHECK-DAG: %[[C12:.*]] = aie.tile(1, 2)
    %c12 = aie.tile(1, 2)
    // CHECK-DAG: %[[C22:.*]] = aie.tile(2, 2)
    %c22 = aie.tile(2, 2)
    // CHECK-DAG: %[[C32:.*]] = aie.tile(3, 2)
    %c32 = aie.tile(3, 2)
    // CHECK-DAG: %[[C42:.*]] = aie.tile(4, 2)
    %c42 = aie.tile(4, 2)
    // CHECK-DAG: %[[C52:.*]] = aie.tile(5, 2)
    %c52 = aie.tile(5, 2)
    // CHECK-DAG: %[[C62:.*]] = aie.tile(6, 2)
    %c62 = aie.tile(6, 2)
    // CHECK-DAG: %[[C72:.*]] = aie.tile(7, 2)
    %c72 = aie.tile(7, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(0, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)

    // Four single-dest flows all pointing at col 0 (3 packet + 1 circuit).
    aie.packet_flow(0x1) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c04, DMA : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c03, DMA : 0>
    }
    aie.packet_flow(0x3) {
      aie.packet_source<%shim, DMA : 1>
      aie.packet_dest<%c02, DMA : 1>
    }
    aie.flow(%c03, DMA : 0, %shim, DMA : 0)

    // One 8-way packet broadcast over cols 0..7 (span midpoint = 3.5).
    aie.packet_flow(0x4) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c02, DMA : 0>
      aie.packet_dest<%c12, DMA : 0>
      aie.packet_dest<%c22, DMA : 0>
      aie.packet_dest<%c32, DMA : 0>
      aie.packet_dest<%c42, DMA : 0>
      aie.packet_dest<%c52, DMA : 0>
      aie.packet_dest<%c62, DMA : 0>
      aie.packet_dest<%c72, DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Pure broadcast: cost is flat in [1, 7]; tiebreak picks the span mid
// (col 4).

// CHECK-LABEL: @shim_pure_broadcast_tiebreaks_to_span_mid
module @shim_pure_broadcast_tiebreaks_to_span_mid {
  aie.device(npu2) {
    // CHECK-DAG: %[[C1:.*]] = aie.tile(1, 2)
    %c1 = aie.tile(1, 2)
    // CHECK-DAG: %[[C7:.*]] = aie.tile(7, 2)
    %c7 = aie.tile(7, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(4, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.packet_flow(0x1) {
      aie.packet_source<%shim, DMA : 0>
      aie.packet_dest<%c1, DMA : 0>
      aie.packet_dest<%c7, DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// packet_flow peers are sources x destinations, not destinations x
// destinations. A memtile that's one of three destinations of a
// broadcast follows the source's column, not the other dests'. Memtile
// here is a dest along with two cores at cols 2 and 6, sourced from col
// 0 -- memtile must land at col 0, not col 4.

// CHECK-LABEL: @memtile_dest_peers_are_sources_only
module @memtile_dest_peers_are_sources_only {
  aie.device(npu2) {
    // CHECK-DAG: %[[SRC:.*]] = aie.tile(0, 2)
    %src = aie.tile(0, 2)
    // CHECK-DAG: %[[D2:.*]] = aie.tile(2, 2)
    %d2 = aie.tile(2, 2)
    // CHECK-DAG: %[[D6:.*]] = aie.tile(6, 2)
    %d6 = aie.tile(6, 2)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(0, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    aie.packet_flow(0x1) {
      aie.packet_source<%src, DMA : 0>
      aie.packet_dest<%d2, DMA : 0>
      aie.packet_dest<%d6, DMA : 0>
      aie.packet_dest<%mem, DMA : 0>
    }
    // CHECK-NOT: aie.logical_tile
  }
}

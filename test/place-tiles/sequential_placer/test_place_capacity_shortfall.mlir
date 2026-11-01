//===- test_place_capacity_shortfall.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ASYM

// All eight MemTiles have one output channel free: eight free channels in
// total cannot accommodate a logical tile that needs two on the same tile.
module @npu2_memtile_partial_bookings {
  aie.device(npu2) {
    %mem0 = aie.logical_tile<MemTile>(0, 1)
    %mem1 = aie.logical_tile<MemTile>(1, 1)
    %mem2 = aie.logical_tile<MemTile>(2, 1)
    %mem3 = aie.logical_tile<MemTile>(3, 1)
    %mem4 = aie.logical_tile<MemTile>(4, 1)
    %mem5 = aie.logical_tile<MemTile>(5, 1)
    %mem6 = aie.logical_tile<MemTile>(6, 1)
    %mem7 = aie.logical_tile<MemTile>(7, 1)
    %core = aie.tile(0, 2)

    // CHECK: error: no MemTile on the device has 0 input/2 output DMA channel(s) free: all 8 MemTile(s) are at 0/48 input, 40/48 output channels used
    // CHECK: note: the closest MemTile is short 1 output channel(s); fan traffic through a MemTile via aie.objectfifo.link
    %mem = aie.logical_tile<MemTile>(?, ?)

    aie.packet_flow(0x1) {
      aie.packet_source<%mem0, DMA : 0>
      aie.packet_source<%mem0, DMA : 1>
      aie.packet_source<%mem0, DMA : 2>
      aie.packet_source<%mem0, DMA : 3>
      aie.packet_source<%mem0, DMA : 4>
      aie.packet_source<%mem1, DMA : 0>
      aie.packet_source<%mem1, DMA : 1>
      aie.packet_source<%mem1, DMA : 2>
      aie.packet_source<%mem1, DMA : 3>
      aie.packet_source<%mem1, DMA : 4>
      aie.packet_source<%mem2, DMA : 0>
      aie.packet_source<%mem2, DMA : 1>
      aie.packet_source<%mem2, DMA : 2>
      aie.packet_source<%mem2, DMA : 3>
      aie.packet_source<%mem2, DMA : 4>
      aie.packet_source<%mem3, DMA : 0>
      aie.packet_source<%mem3, DMA : 1>
      aie.packet_source<%mem3, DMA : 2>
      aie.packet_source<%mem3, DMA : 3>
      aie.packet_source<%mem3, DMA : 4>
      aie.packet_source<%mem4, DMA : 0>
      aie.packet_source<%mem4, DMA : 1>
      aie.packet_source<%mem4, DMA : 2>
      aie.packet_source<%mem4, DMA : 3>
      aie.packet_source<%mem4, DMA : 4>
      aie.packet_source<%mem5, DMA : 0>
      aie.packet_source<%mem5, DMA : 1>
      aie.packet_source<%mem5, DMA : 2>
      aie.packet_source<%mem5, DMA : 3>
      aie.packet_source<%mem5, DMA : 4>
      aie.packet_source<%mem6, DMA : 0>
      aie.packet_source<%mem6, DMA : 1>
      aie.packet_source<%mem6, DMA : 2>
      aie.packet_source<%mem6, DMA : 3>
      aie.packet_source<%mem6, DMA : 4>
      aie.packet_source<%mem7, DMA : 0>
      aie.packet_source<%mem7, DMA : 1>
      aie.packet_source<%mem7, DMA : 2>
      aie.packet_source<%mem7, DMA : 3>
      aie.packet_source<%mem7, DMA : 4>
      aie.packet_source<%mem, DMA : 4>
      aie.packet_source<%mem, DMA : 5>
      aie.packet_dest<%core, DMA : 0>
    }
  }
}

// -----

// For a 3 input/3 output request, the first tile is short 3 input/0 output,
// the second 1 input/1 output. Choose the second pair (sum 2 < 3), not the
// independent directional minima (1 input/0 output).
module @closest_memtile_joint_shortfall {
  aie.device(npu2_2col) {
    %mem0 = aie.logical_tile<MemTile>(0, 1)
    %mem1 = aie.logical_tile<MemTile>(1, 1)
    %core = aie.tile(0, 2)

    // ASYM: error: no MemTile on the device has 3 input/3 output DMA channel(s) free: all 2 MemTile(s) are at 10/12 input, 6/12 output channels used
    // ASYM: note: the closest MemTile is short 1 input/1 output channel(s); fan traffic through a MemTile via aie.objectfifo.link
    %mem = aie.logical_tile<MemTile>(?, ?)

    aie.packet_flow(0x1) {
      aie.packet_source<%core, DMA : 0>
      aie.packet_dest<%mem0, DMA : 0>
      aie.packet_dest<%mem0, DMA : 1>
      aie.packet_dest<%mem0, DMA : 2>
      aie.packet_dest<%mem0, DMA : 3>
      aie.packet_dest<%mem0, DMA : 4>
      aie.packet_dest<%mem0, DMA : 5>
      aie.packet_dest<%mem1, DMA : 0>
      aie.packet_dest<%mem1, DMA : 1>
      aie.packet_dest<%mem1, DMA : 2>
      aie.packet_dest<%mem1, DMA : 3>
      aie.packet_dest<%mem, DMA : 3>
      aie.packet_dest<%mem, DMA : 4>
      aie.packet_dest<%mem, DMA : 5>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%mem0, DMA : 0>
      aie.packet_source<%mem0, DMA : 1>
      aie.packet_source<%mem1, DMA : 0>
      aie.packet_source<%mem1, DMA : 1>
      aie.packet_source<%mem1, DMA : 2>
      aie.packet_source<%mem1, DMA : 3>
      aie.packet_source<%mem, DMA : 3>
      aie.packet_source<%mem, DMA : 4>
      aie.packet_source<%mem, DMA : 5>
      aie.packet_dest<%core, DMA : 0>
    }
  }
}

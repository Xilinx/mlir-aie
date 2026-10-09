//===- coverage_reserved_arbiters.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s

// The four flows through memtile (0,1) can deadlock one another pairwise (see
// arbiter_deadlock_exhausted.mlir), so each needs an arbiter of its own. An
// arbiter whose four msels an existing masterset already uses is not free.

// Three arbiters taken leaves three for four flows.

// CHECK: error: Unable to find a legal routing: at tile (0, 1),
// CHECK-SAME: but the switchbox has 3 free.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
      %a20 = aie.amsel<2> (0)
      %a21 = aie.amsel<2> (1)
      %a22 = aie.amsel<2> (2)
      %a23 = aie.amsel<2> (3)
      aie.masterset(North : 2, %a20, %a21, %a22, %a23)
    }
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%m1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%m1, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%m1, DMA : 2> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
  }
}

// -----

// Two taken leaves four, and the flows keep off the taken ones.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.masterset(North : 1,
// CHECK-NOT:     aie.amsel<{{[01]}}>
// CHECK-COUNT-4: aie.amsel<{{[2-5]}}> (0)
// CHECK-NOT:     aie.amsel
// CHECK:       }

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %sb = aie.switchbox(%m) {
      %a00 = aie.amsel<0> (0)
      %a01 = aie.amsel<0> (1)
      %a02 = aie.amsel<0> (2)
      %a03 = aie.amsel<0> (3)
      aie.masterset(North : 0, %a00, %a01, %a02, %a03)
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      %a13 = aie.amsel<1> (3)
      aie.masterset(North : 1, %a10, %a11, %a12, %a13)
    }
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%m1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%m1, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%m1, DMA : 2> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
  }
}

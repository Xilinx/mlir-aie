//===- arbiter_deadlock_exhausted.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Companion to arbiter_deadlock_avoidance.mlir: the hazard cannot be avoided.
// All six arbiters at memtile (0,1) carry a flow its DMA produces, so the
// seventh flow -- consumed by that same DMA -- depends on whichever arbiter it
// gets. Routing succeeds unchanged, and warns.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    // Six flows out of the memtile, one per DMA channel, each leaving the
    // switchbox on a different master port and so taking its own arbiter.
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%c4, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%c5, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%s0, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%s1, DMA : 0> }

    // The seventh flow, back into that memtile.
    aie.packet_flow(6) { aie.packet_source<%s2, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
  }
}

// WARN: warning: packet flow 6 shares arbiter 0 with a flow it can deadlock against

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[SHARED:.*]] = aie.amsel<0> (1)
// CHECK:         aie.masterset(DMA : 0, %[[SHARED]])

//===- arbiter_deadlock_avoidance.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Memtile (0,1) is a relay: flow 6 arrives at DMA : 1, flow 0 leaves from
// DMA : 0. Nothing programs the DMAs, so each channel is assumed to wait on
// anything on its tile, and flow 0 can deadlock against flow 6 and against
// every flow into or out of shim (0,0) and memtile (0,1). Routed the shortest
// way, seven flows cross (0,1) and two of them would have to share an arbiter
// with flow 0, so the router takes one of the others around (0,1) instead.

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t10 = aie.tile(1, 0)
    %t20 = aie.tile(2, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    // The relay pair through memtile (0,1): flow 0 out, flow 6 in.
    aie.packet_flow(0) { aie.packet_source<%t01, DMA : 0>  aie.packet_dest<%t02, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%t03, DMA : 1>  aie.packet_dest<%t01, DMA : 1> }

    // Filler, to crowd (0,1).
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0>  aie.packet_dest<%t00, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t00, DMA : 1>  aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%t10, DMA : 0>  aie.packet_dest<%t05, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%t10, DMA : 1>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(7) { aie.packet_source<%t20, DMA : 0>  aie.packet_dest<%t05, DMA : 1> }
    aie.packet_flow(8) { aie.packet_source<%t04, DMA : 1>  aie.packet_dest<%t00, DMA : 1> }
  }
}

// Six flows are left at (0,1), each with an arbiter of its own.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-COUNT-6: aie.amsel<{{[0-5]}}> (0)
// CHECK-NOT:     aie.amsel
// CHECK:         aie.masterset(DMA : 1,
// CHECK:         aie.rule(31, 6,
// CHECK:         aie.rule(31, 0,
// CHECK:       }

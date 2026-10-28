//===- arbiter_broadcast_circuit_hops.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=PINNED

// Six cores each feed their own S2MM channel of memtile (0,1), and nothing
// says how much, so any two of those flows can deadlock on a shared arbiter.
// Memtile MM2S 0 broadcasts to core (0,3) and memtile (1,1) -- eight packet
// master ports at (0,1) for six arbiters. The broadcast leaves (0,1) on one
// master port and branches further on; it alone uses that slave and master
// port, so the hop becomes aie.connect, leaving each join flow an arbiter of
// its own. Without that, the broadcast takes an arbiter at (0,1) too, and the
// router fails.

// NOWARN-NOT: warning

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-NEXT:    aie.connect<DMA : 0, {{North|South|East|West}} : {{[0-9]+}}>
// CHECK-NEXT:    aie.amsel<0> (0)
// CHECK-NEXT:    aie.amsel<1> (0)
// CHECK-NEXT:    aie.amsel<2> (0)
// CHECK-NEXT:    aie.amsel<3> (0)
// CHECK-NEXT:    aie.amsel<4> (0)
// CHECK-NEXT:    aie.amsel<5> (0)
// CHECK-NOT:     aie.amsel
// CHECK:       aie.switchbox

// PINNED: error: Unable to find a legal routing: at tile (0, 1), no two of
// PINNED-SAME: can share an arbiter, and each takes one there whatever the routing, but the switchbox has 6 free.

module {
  aie.device(npu2) {
    %m  = aie.tile(0, 1)
    %m1 = aie.tile(1, 1)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(1, 2)
    %t2 = aie.tile(2, 2)
    %t3 = aie.tile(3, 2)
    %t4 = aie.tile(4, 2)
    %t5 = aie.tile(5, 2)
    %t6 = aie.tile(0, 3)
    aie.packet_flow(0) { aie.packet_source<%t0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%t2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(3) { aie.packet_source<%t3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(4) { aie.packet_source<%t4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(5) { aie.packet_source<%t5, DMA : 0> aie.packet_dest<%m, DMA : 5> }
    aie.packet_flow(6) {
      aie.packet_source<%m, DMA : 0>
      aie.packet_dest<%t6, DMA : 1>
      aie.packet_dest<%m1, DMA : 0>
    }
  }
}

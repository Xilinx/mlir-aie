//===- arbiter_port_stall_aggregation.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Memtile (0,1) sends flows 0 and 1 from one DMA channel -- flow 0 to a core's
// DMA, flow 1 as control packets to the shim -- on different master ports, and
// so different arbiters. One slave port is one ordered stream, so flow 1 sits
// behind flow 0 and stops whenever the core stops draining. But it stops
// between packets: an arbiter is held from a packet's header to its tlast, and
// a packet still queued behind another holds none. Flow 6, placed last and on
// its way south, has to share an arbiter, and none of the flows here can
// deadlock against it -- nothing they feed waits on (0,5) or on shim (0,0)'s
// S2MM -- so it takes the first arbiter, flow 2's, and nothing is reported.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s0, TileControl : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%c4, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%c2, DMA : 1> }

    // Ends at this memtile's DMA.
    aie.packet_flow(5) { aie.packet_source<%c4, DMA : 0>  aie.packet_dest<%m, DMA : 0> }

    // Passes through (0,1) on its way south. Placed last, so it has to share.
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
  }
}

// NOWARN-NOT: warning

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FIRST:.*]] = aie.amsel<0> (0)
// CHECK:         %[[CTRL:.*]] = aie.amsel<5> (0)
// CHECK:         %[[SHARED:.*]] = aie.amsel<0> (1)
// CHECK:         aie.masterset(South : 2, %[[CTRL]])
// CHECK:         aie.masterset(South : 3, %[[SHARED]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 6, %[[SHARED]])
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 2, %[[FIRST]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 1, %[[CTRL]])

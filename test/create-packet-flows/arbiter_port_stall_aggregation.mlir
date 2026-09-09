//===- arbiter_port_stall_aggregation.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Memtile (0,1) sends flows 0 and 1 from one DMA channel: flow 0 feeds a core's
// DMA, flow 1 carries control packets to the shim. They take different master
// ports, so they take different arbiters.
//
// A slave port carries one ordered stream, so flow 1 sits behind flow 0 and
// stops whenever the core stops draining. Its arbiter is held for as long as
// that lasts, so flow 6, which has yet to leave this switchbox, must not join
// it. Flow 5 ends at a DMA here and is drained locally, so flow 6 joins that.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s0, TileControl : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%c4, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%c5, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%c2, DMA : 1> }

    // Ends at this memtile's DMA.
    aie.packet_flow(5) { aie.packet_source<%c4, DMA : 0>  aie.packet_dest<%m, DMA : 0> }

    // Passes through (0,1) on its way south. Placed last, so it has to share.
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
  }
}

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[CTRL:.*]] = aie.amsel<1> (0)
// CHECK:         %[[ENDSHERE:.*]] = aie.amsel<5> (0)
// CHECK:         %[[SHARED:.*]] = aie.amsel<5> (1)
// CHECK:         aie.masterset(DMA : 0, %[[ENDSHERE]])
// CHECK:         aie.masterset(South : 2, %[[CTRL]])
// CHECK:         aie.masterset(South : 3, %[[SHARED]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 6, %[[SHARED]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 1, %[[CTRL]])

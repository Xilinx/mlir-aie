//===- arbiter_shared_slave_port.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Two flows on one slave port never arbitrate against each other: the port
// hands the switch one stream, and the arbiter sees the second flow only once
// the first has finished. So they may share an arbiter even when the port they
// arrive on is one that stalls.
//
// Flows 0..4 fill msel 0 on arbiters 0..4. Flows 5 and 6 then leave DMA : 5
// together, southbound and northbound, so they need separate master ports and
// separate amsels. Flow 5 takes the last free arbiter. Flow 6 finds every
// arbiter occupied, and takes the second msel of the one carrying its own
// slave port rather than crowding a stranger.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %s4 = aie.tile(4, 0)
    %s5 = aie.tile(5, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)

    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%s2, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%s3, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s4, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%s5, DMA : 0> }

    // Both out of DMA : 5, in opposite directions.
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%s0, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%c5, TileControl : 0> }
  }
}

// NOWARN-NOT: warning

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[SOUTH:.*]] = aie.amsel<5> (0)
// CHECK:         %[[NORTH:.*]] = aie.amsel<5> (1)
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule(31, 6, %[[NORTH]])
// CHECK-NEXT:      aie.rule(31, 5, %[[SOUTH]])

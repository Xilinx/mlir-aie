//===- arbiter_multi_hop_avoidance.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=QUIET --allow-empty

// The same three-stage pipeline as arbiter_multi_hop_cycle.mlir -- flow 0 into
// core (0,4), flow 1 on to (0,5), flow 2 back out to a shim, with flows 0 and 2
// meeting at memtile (0,1). There nothing safe was left and the cycle could
// only be reported; here flow 8 gives the allocator a clean arbiter to find.
//
// Flow 8 ends at this memtile's own DMA and nothing the memtile sends reaches
// (0,5), so it is coupled to flow 2 in neither direction. Flow 2 must land on
// its arbiter rather than flow 0's, and silently.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    // Filler on this memtile's own DMAs, to use up arbiters 0..3.
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%s2, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s3, DMA : 0> }

    // Ends at this memtile, and uncoupled from flow 2 either way.
    aie.packet_flow(8) { aie.packet_source<%c3, DMA : 1>  aie.packet_dest<%m, DMA : 4> }

    // The pipeline. Flows 0 and 2 are the two ends of it.
    aie.packet_flow(0) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%c4, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%c4, DMA : 0>  aie.packet_dest<%c5, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%c5, DMA : 1>  aie.packet_dest<%s1, DMA : 0> }
  }
}

// Seven groups on six arbiters again, so flow 2 still doubles up -- but on
// arbiter 5 with flow 8, not arbiter 4 with flow 0. Pairwise endpoint checks
// picked arbiter 4: flows 0 and 2 share no endpoint, so it looked clean.

// QUIET-NOT: warning

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[PIPEIN:.*]] = aie.amsel<4> (0)
// CHECK:         %[[ENDSHERE:.*]] = aie.amsel<5> (0)
// CHECK:         %[[PIPEOUT:.*]] = aie.amsel<5> (1)
// CHECK:         aie.masterset(DMA : 4, %[[ENDSHERE]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 2, %[[PIPEOUT]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 0, %[[PIPEIN]])
// CHECK:         aie.packet_rules(North : 2) {
// CHECK-NEXT:      aie.rule(31, 8, %[[ENDSHERE]])

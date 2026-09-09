//===- arbiter_multi_hop_cycle.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// The cycle the arbiter closes need not run between the two flows sharing it.
// Here it runs the long way round a three-stage pipeline:
//
//   flow 0: shim (0,0) --> core (0,4)
//   flow 1: core (0,4) --> core (0,5)
//   flow 2: core (0,5) --> shim (1,0)
//
// Flows 0 and 2 both transit memtile (0,1), on South : 1 and North : 3. Put
// them on one arbiter and: the grant goes to flow 0, core (0,4)'s input buffer
// is full so flow 0 stalls with no tlast, (0,4) can only free that buffer by
// running and emitting flow 1, (0,5) consumes it and emits flow 2 -- which
// wants the arbiter flow 0 is still holding.
//
// None of the pairwise rules sees this. At (0,1) neither flow arrives on a DMA
// of this tile, so neither counts as a port that stalls; flow 0's endpoints and
// flow 2's are two hops apart, so comparing the two flows directly finds
// nothing in common; and there are no DMA bodies to find a send outlasting its
// receive descriptors in. It is only visible as reachability over the whole
// flow graph -- (0,4), then (0,5), then flow 2's producer -- which is what the
// coupling rule walks.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    // The pipeline. Flows 0 and 2 are the two ends of it.
    aie.packet_flow(0) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%c4, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%c4, DMA : 0>  aie.packet_dest<%c5, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%c5, DMA : 1>  aie.packet_dest<%s1, DMA : 0> }

    // Filler on this memtile's own DMAs, to use up the arbiters at (0,1).
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%s0, DMA : 1> }
    aie.packet_flow(6) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s0, DMA : 0> }
    aie.packet_flow(7) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%c2, DMA : 1> }
  }
}

// Seven groups on six arbiters, so flow 2 has to double up somewhere. Every
// arbiter here carries something: 0..4 hold this memtile's own outgoing DMAs,
// which stall whenever the core they feed stops draining, and 5 holds flow 0.
// Only flow 0 closes a cycle, so flow 2 takes arbiter 0 and the warning names
// the merely-stalling flow it settled for.

// WARN: warning: at tile (0, 1), packet flow 2 shares arbiter 0 with packet flow 3, which it can deadlock against

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FILLER:.*]] = aie.amsel<0> (0)
// CHECK:         %[[PIPEIN:.*]] = aie.amsel<5> (0)
// CHECK:         %[[PIPEOUT:.*]] = aie.amsel<0> (1)
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 3, %[[FILLER]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 2, %[[PIPEOUT]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 0, %[[PIPEIN]])

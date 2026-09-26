//===- arbiter_multi_hop_avoidance.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=QUIET --allow-empty

// The same three-stage pipeline as arbiter_multi_hop_cycle.mlir -- flow 0 into
// core (0,4), flow 1 on to (0,5), flow 2 back out to a shim, with flows 0 and 2
// meeting at memtile (0,1) -- plus flow 8, which ends at this memtile's own
// DMA. Nothing the memtile sends reaches (0,5), so flow 8 is coupled to flow 2
// in neither direction. Flow 2 must not land on flow 0's arbiter, and the
// router must stay silent.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %m  = aie.tile(0, 1)
    %m11 = aie.tile(1, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    // No way into shim (1,0) but from shim (0,0): memtile (1,1)'s South master
    // ports and shim (2,0)'s West master ports are taken.
    %sb11 = aie.switchbox(%m11) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<DMA : 1, South : 1>
      aie.connect<DMA : 2, South : 2>
      aie.connect<DMA : 3, South : 3>
    }
    %sb20 = aie.switchbox(%s2) {
      aie.connect<North : 0, West : 0>
      aie.connect<North : 1, West : 1>
      aie.connect<North : 2, West : 2>
      aie.connect<North : 3, West : 3>
    }

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

// Seven groups on six arbiters again, so flow 2 still doubles up -- but not
// with flow 0. Pairwise endpoint checks picked flow 0's arbiter: flows 0 and 2
// share no endpoint, so it looked clean. It lands with filler flow 4 instead,
// which ends at core (0,3): nothing there waits on the pipeline, and nothing
// at shim (1,0) waits on flow 4.

// QUIET-NOT: warning

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FILLER:.*]] = aie.amsel<0> (0)
// CHECK:         %[[PIPEIN:.*]] = aie.amsel<5> (0)
// CHECK:         %[[PIPEOUT:.*]] = aie.amsel<0> (1)
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 2, %[[PIPEOUT]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 0, %[[PIPEIN]])
// CHECK:         aie.packet_rules(DMA : 1) {
// CHECK-NEXT:      aie.rule(31, 4, %[[FILLER]])

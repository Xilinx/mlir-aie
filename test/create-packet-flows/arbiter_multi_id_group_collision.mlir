//===- arbiter_multi_id_group_collision.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Six shim channels each send ids 0..2 into their own memtile (0,1) S2MM
// channel. The ids fold into one rule (mask 28), so each group is three
// streams behind one arbiter, and the six groups fill the memtile's six
// arbiters. A seventh group, ids 4..6, passes (0,1) on its way to core (0,3).
// Giving it a packet master there would put two groups on one arbiter behind
// a single hold-until-tlast grant, which stalls on NPU2 hardware after a few
// iterations.
//
// The router circuit-switches the transit hop, or without circuit hops routes
// the group around (0,1), leaving each DMA group its own arbiter.

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s --check-prefix=NOCIRCUIT

// CHECK-NOT:     warning
// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-NEXT:    aie.connect<South : 1, North : 1>
// CHECK-DAG:     %[[A0:.*]] = aie.amsel<0> (0)
// CHECK-DAG:     %[[A1:.*]] = aie.amsel<1> (0)
// CHECK-DAG:     %[[A2:.*]] = aie.amsel<2> (0)
// CHECK-DAG:     %[[A3:.*]] = aie.amsel<3> (0)
// CHECK-DAG:     %[[A4:.*]] = aie.amsel<4> (0)
// CHECK-DAG:     %[[A5:.*]] = aie.amsel<5> (0)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[A2]])
// CHECK-DAG:     aie.masterset(DMA : 1, %[[A3]])
// CHECK-DAG:     aie.masterset(DMA : 2, %[[A1]])
// CHECK-DAG:     aie.masterset(DMA : 3, %[[A4]])
// CHECK-DAG:     aie.masterset(DMA : 4, %[[A0]])
// CHECK-DAG:     aie.masterset(DMA : 5, %[[A5]])
// CHECK-NOT:     aie.masterset
// CHECK:       aie.switchbox(%tile_0_3)

// NOCIRCUIT-NOT:     warning
// NOCIRCUIT-LABEL: aie.switchbox(%mem_tile_0_1)
// NOCIRCUIT-NOT:     aie.connect
// NOCIRCUIT-DAG:     %[[A0:.*]] = aie.amsel<0> (0)
// NOCIRCUIT-DAG:     %[[A1:.*]] = aie.amsel<1> (0)
// NOCIRCUIT-DAG:     %[[A2:.*]] = aie.amsel<2> (0)
// NOCIRCUIT-DAG:     %[[A3:.*]] = aie.amsel<3> (0)
// NOCIRCUIT-DAG:     %[[A4:.*]] = aie.amsel<4> (0)
// NOCIRCUIT-DAG:     %[[A5:.*]] = aie.amsel<5> (0)
// NOCIRCUIT-DAG:     aie.masterset(DMA : 0, %[[A2]])
// NOCIRCUIT-DAG:     aie.masterset(DMA : 1, %[[A1]])
// NOCIRCUIT-DAG:     aie.masterset(DMA : 2, %[[A3]])
// NOCIRCUIT-DAG:     aie.masterset(DMA : 3, %[[A4]])
// NOCIRCUIT-DAG:     aie.masterset(DMA : 4, %[[A5]])
// NOCIRCUIT-DAG:     aie.masterset(DMA : 5, %[[A0]])
// NOCIRCUIT-NOT:     aie.masterset
// NOCIRCUIT:       aie.switchbox(%tile_0_3)

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %s4 = aie.tile(4, 0)
    %s5 = aie.tile(5, 0)
    %m  = aie.tile(0, 1)
    %c3 = aie.tile(0, 3)
    aie.packet_flow(0) { aie.packet_source<%s0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%s0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%s0, DMA : 0> aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(0) { aie.packet_source<%s1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(1) { aie.packet_source<%s1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(2) { aie.packet_source<%s1, DMA : 0> aie.packet_dest<%m, DMA : 1> }
    aie.packet_flow(0) { aie.packet_source<%s2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(1) { aie.packet_source<%s2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(2) { aie.packet_source<%s2, DMA : 0> aie.packet_dest<%m, DMA : 2> }
    aie.packet_flow(0) { aie.packet_source<%s3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(1) { aie.packet_source<%s3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(2) { aie.packet_source<%s3, DMA : 0> aie.packet_dest<%m, DMA : 3> }
    aie.packet_flow(0) { aie.packet_source<%s4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(1) { aie.packet_source<%s4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(2) { aie.packet_source<%s4, DMA : 0> aie.packet_dest<%m, DMA : 4> }
    aie.packet_flow(0) { aie.packet_source<%s5, DMA : 0> aie.packet_dest<%m, DMA : 5> }
    aie.packet_flow(1) { aie.packet_source<%s5, DMA : 0> aie.packet_dest<%m, DMA : 5> }
    aie.packet_flow(2) { aie.packet_source<%s5, DMA : 0> aie.packet_dest<%m, DMA : 5> }
    aie.packet_flow(4) { aie.packet_source<%s0, DMA : 1> aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%s0, DMA : 1> aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 1> aie.packet_dest<%c3, DMA : 0> }
  }
}

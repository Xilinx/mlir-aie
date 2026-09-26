//===- packet_msel_exhaustion_detour.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Five ids leave (0,2) DMA:0 for five different sets of master ports, all
// holding Core:0. Routed straight, they need one arbiter with five msels, and
// an arbiter has four; this used to crash. Sending all five up to (0,3) and back
// down to Core:0 splits them over two arbiters at (0,2): one for the ports
// DMA:0 feeds, one for those North:0 feeds.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK-DAG:     %[[A0:.*]] = aie.amsel<0> (0)
// CHECK-DAG:     %[[B0:.*]] = aie.amsel<1> (0)
// CHECK-DAG:     %[[A1:.*]] = aie.amsel<0> (1)
// CHECK-DAG:     %[[B1:.*]] = aie.amsel<1> (1)
// CHECK-DAG:     aie.masterset(Core : 0, %[[B0]], %[[B1]])
// CHECK-DAG:     aie.masterset(DMA : 0, %[[A1]])
// CHECK-DAG:     aie.masterset(DMA : 1, %[[B1]])
// CHECK-DAG:     aie.masterset(North : 0, %[[A0]], %[[A1]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK:         aie.masterset(South : 0,

module {
  aie.device(npu1_1col) {
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> }
    aie.packet_flow(2) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(4) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(5) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_3, DMA : 0> }
  }
}

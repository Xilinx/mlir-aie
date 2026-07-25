//===- find_flows_fanout_packet.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A packet broadcast is pinned by keeping its fan-out node materialized (the
// shared amsel driving two mastersets) and lifting the straight-line sections
// around it as packet flows that carry vias. --aie-split-flow-vias rebuilds the
// section switchbox rules, so create-pathfinder-flows reproduces the routing.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | aie-opt --aie-split-flow-vias | aie-opt --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUNDTRIP

// The fan-out node stays materialized: one amsel feeding two mastersets.
// CHECK: %[[P04:.*]] = aie.tile(0, 4)
// CHECK: aie.switchbox(%[[P04]]) {
// CHECK:   %[[AS:.*]] = aie.amsel<0> (0)
// CHECK:   aie.masterset(DMA : 0, %[[AS]])
// CHECK:   aie.masterset(North : {{[0-9]+}}, %[[AS]])
// CHECK:   aie.packet_rules(South : {{[0-9]+}}) {
// CHECK:     aie.rule(31, 1, %[[AS]])
// CHECK:   }
// CHECK: }
// The sections into and out of the fan-out node are pinned packet flows.
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%{{.*}}, DMA : 0>
// CHECK:   aie.packet_dest<%[[P04]], South : {{[0-9]+}}>
// CHECK: } via (
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%[[P04]], North : {{[0-9]+}}>
// CHECK:   aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK: } via (

// After splitting the vias and re-routing, the broadcast is reproduced: one
// amsel drives both the local DMA and the northbound continuation.
// ROUNDTRIP: aie.masterset(DMA : 0, %[[R:.*]])
// ROUNDTRIP: aie.masterset(North : {{[0-9]+}}, %[[R]])
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(0x1) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
      aie.packet_dest<%t05, DMA : 0>
    }
  }
}

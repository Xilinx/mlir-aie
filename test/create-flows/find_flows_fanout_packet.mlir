//===- find_flows_fanout_packet.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A packet broadcast is split at its fan-out node: the shared amsel and its two
// mastersets stay explicit (the broadcast is no longer implicit), and the
// sections around it become logical packet flows.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | FileCheck %s

// CHECK: %[[P04:.*]] = aie.tile(0, 4)
// CHECK: aie.switchbox(%[[P04]]) {
// CHECK:   %[[AS:.*]] = aie.amsel<0> (0)
// CHECK:   aie.masterset(DMA : 0, %[[AS]])
// CHECK:   aie.masterset(North : {{[0-9]+}}, %[[AS]])
// CHECK:   aie.packet_rules(South : {{[0-9]+}}) {
// CHECK:     aie.rule(31, 1, %[[AS]])
// CHECK:   }
// CHECK: }
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%{{.*}}, DMA : 0>
// CHECK:   aie.packet_dest<%[[P04]], South : {{[0-9]+}}>
// CHECK: }
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%[[P04]], North : {{[0-9]+}}>
// CHECK:   aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK: }
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

//===- find_flows_fanout_packet.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// In pinning mode (emit-vias) packet routes are held as their exact physical
// configuration: every switchbox -- including the fan-out node whose shared
// amsel drives two mastersets -- is kept verbatim, and the aie.packet_flow op
// is dropped so re-routing preserves the pinned route instead of re-deriving it.

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
// CHECK-NOT: aie.packet_flow
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

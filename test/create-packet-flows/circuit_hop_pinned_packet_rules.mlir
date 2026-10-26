//===- circuit_hop_pinned_packet_rules.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Packet id 10 alone passes (1,3), entering on South:4. That makes the hop a
// candidate for a circuit, but the switchbox already has packet rules on
// South:4, so the hop stays packet-switched and joins them. Reduced from
// router_mutation.py seed 190.

// CHECK-LABEL: aie.switchbox(%tile_1_3)
// CHECK-NOT:     aie.connect<South : 4
// CHECK:         aie.packet_rules(South : 4) {
// CHECK-NEXT:      aie.rule(31, 5,
// CHECK-NEXT:      aie.rule(31, 10,

module {
  aie.device(npu2_3col) {
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %sb_1_3 = aie.switchbox(%t_1_3) {
      %a4_0 = aie.amsel<4> (0)
      aie.packet_rules(South : 4) {
        aie.rule(31, 5, %a4_0)
      }
    }
    aie.packet_flow(28) {
      aie.packet_source<%t_1_1, DMA : 1>
      aie.packet_dest<%t_1_2, DMA : 1>
      aie.packet_dest<%t_1_3, DMA : 0>
      aie.packet_dest<%t_1_4, DMA : 1>
    }
    aie.packet_flow(14) {
      aie.packet_source<%t_1_2, DMA : 1>
      aie.packet_dest<%t_1_3, DMA : 1>
      aie.packet_dest<%t_1_5, Core : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%t_1_4, Core : 0>
      aie.packet_dest<%t_1_2, DMA : 1>
      aie.packet_dest<%t_1_3, DMA : 1>
    }
    aie.packet_flow(10) {
      aie.packet_source<%t_1_1, DMA : 5>
      aie.packet_dest<%t_1_5, DMA : 0>
    }
  }
}

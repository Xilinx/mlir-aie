//===- arbitration.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packet-flow labels include the matching arbitration rule and packet
// metadata. Only the rule that claims the arriving packet is annotated: the
// flow carries 2, which rule 1 claims and rule 0 does not.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot %s | FileCheck %s

// CHECK: arbitration_0 {{.*}} label="r1 pkt=2/30 -> A1.M1 header ctrl-overlay"
// CHECK-SAME: tooltip="rule 1: packet ID & 30 = 2, arbiter 1, msel 1, keep packet header: yes, priority route: no, control overlay: yes"
// CHECK-SAME: class="arbitration detail-arbitration detail-vias flow src-0 dst-0"
// CHECK-NOT: arbitration_1
// The port a packet crosses is named for that packet, so two flows sharing
// one port stay apart.
// CHECK: p_0_2_5_0_k2 -> p_0_3_3_0_k2
// CHECK-SAME: F0 pkt=2/30 header priority

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    aie.packet_flow(2, mask = 30) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    } via (%t02 : DMA : 0 -> North : 0,
           %t03 : South : 0 -> DMA : 0) {
      keep_pkt_header = true,
      priority_route = true
    }

    aie.switchbox(%t02) {
      %a10 = aie.amsel<1> (0)
      %a11 = aie.amsel<1> (1)
      %a12 = aie.amsel<1> (2)
      aie.masterset(North : 0, %a10, %a11, %a12) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a10) {aie.priority_route}
        aie.rule(30, 2, %a11)
        aie.rule(31, 2, %a12)
      }
    }
  }
}

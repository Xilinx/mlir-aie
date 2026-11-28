//===- arbitration.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packet-flow labels include the matching arbitration rule and packet metadata.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --no-follow-buffers %s | FileCheck %s

// CHECK: arbitration_0 {{.*}} label="r0 pkt=3/31 -> A1.M0 header priority ctrl-overlay"
// CHECK-SAME: tooltip="rule 0: packet ID & 31 = 3, arbiter 1, msel 0, keep packet header: yes, priority route: yes, control overlay: yes"
// CHECK-SAME: class="arbitration detail-arbitration detail-vias flow flow-0"
// CHECK: arbitration_1 {{.*}} label="r1 pkt=2/30 -> A1.M1 header ctrl-overlay"
// CHECK-SAME: class="arbitration detail-arbitration detail-vias flow flow-0"
// CHECK-NOT: arbitration_2
// CHECK: p_0_2_5_0 -> p_0_3_3_0
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

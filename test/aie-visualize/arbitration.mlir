//===- arbitration.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packet-flow labels include the matching arbitration rule and packet
// metadata. The flow is declared with mask 30, so it claims every ID that
// agrees with 2 on the bits that mask selects: 2 and 3. Rule 1 claims 2 and
// rule 0 claims 3, so both rules are annotated and both packets reach the
// channel at the far end.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --split-packet-ports --control-overlay %s | FileCheck %s
// RUN: aie-visualize --emit-dot --split-packet-ports %s | FileCheck %s --check-prefix=NO-OVERLAY

// CHECK-DAG: arbitration_0 {{.*}} label="r0 pkt=3/31 -> A1.M0 header priority ctrl-overlay"
// CHECK-DAG: arbitration_1 {{.*}} label="r1 pkt=2/30 -> A1.M1 header ctrl-overlay"
// CHECK-DAG: tooltip="rule 1: packet ID & 30 = 2, arbiter 1, msel 1, keep packet header: yes, priority route: no, control overlay: yes"
// The port a packet crosses is named for that packet, so the two the flow
// carries stay apart all the way to the channel they share.
// CHECK-DAG: p_0_2_5_0_k2 -> p_0_3_3_0_k2
// CHECK-DAG: p_0_2_5_0_k3 -> p_0_3_3_0_k3
// CHECK-DAG: p_0_3_3_0_k2 -> p_0_3_1_0_s
// CHECK-DAG: p_0_3_3_0_k3 -> p_0_3_1_0_s

// The rules here all carry aie.is_ctrl_pkt_overlay, so without
// --control-overlay none of them is read. The flow itself still is.
// NO-OVERLAY-NOT: arbitration_
// NO-OVERLAY: p_0_2_5_0_k2 -> p_0_3_3_0_k2

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

//===- split_packet_flow_vias_shared_egress.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-split-flow-vias %s | aie-opt | FileCheck %s

// Two packet IDs that use the same pinned egress share one masterset. Its
// AMSels use one arbiter because a hardware master port can select only one
// arbiter.

// CHECK: %[[VIA:.*]] = aie.tile(0, 3)
// CHECK: aie.switchbox(%[[VIA]]) {
// CHECK-DAG: %[[M0:.*]] = aie.amsel<0> (0)
// CHECK-DAG: %[[M1:.*]] = aie.amsel<0> (1)
// CHECK: aie.masterset(DMA : 0, %[[M0]], %[[M1]])
// CHECK: aie.packet_rules(South : 0) {
// CHECK:   aie.rule(31, 1, %[[M0]])
// CHECK: }
// CHECK: aie.packet_rules(South : 1) {
// CHECK:   aie.rule(31, 2, %[[M1]])
// CHECK: }
// CHECK: }
// CHECK-NOT: aie.masterset(DMA : 0

module {
  aie.device(npu1_1col) {
    %src = aie.tile(0, 2)
    %via = aie.tile(0, 3)
    aie.packet_flow(1) {
      aie.packet_source<%src, DMA : 0>
      aie.packet_dest<%via, DMA : 0>
    } via (%via : South : 0 -> DMA : 0)
    aie.packet_flow(2) {
      aie.packet_source<%src, DMA : 1>
      aie.packet_dest<%via, DMA : 0>
    } via (%via : South : 1 -> DMA : 0)
  }
}
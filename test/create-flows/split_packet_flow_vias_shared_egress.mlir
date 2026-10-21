//===- split_packet_flow_vias_shared_egress.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-split-flow-vias %s | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --aie-split-flow-vias --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=ROUTED

// Each pinned hop becomes a local packet flow. The router assigns one AMSel
// and masterset to the shared egress.

// SPLIT: %[[SRC:.*]] = aie.tile(0, 2)
// SPLIT: %[[VIA:.*]] = aie.tile(0, 3)
// SPLIT: aie.packet_flow(1) {
// SPLIT:   aie.packet_source<%[[SRC]], DMA : 0>
// SPLIT:   aie.packet_dest<%[[SRC]], North : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(1) {
// SPLIT:   aie.packet_source<%[[VIA]], South : 0>
// SPLIT:   aie.packet_dest<%[[VIA]], DMA : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(2) {
// SPLIT:   aie.packet_source<%[[SRC]], DMA : 1>
// SPLIT:   aie.packet_dest<%[[SRC]], North : 1>
// SPLIT: }
// SPLIT: aie.packet_flow(2) {
// SPLIT:   aie.packet_source<%[[VIA]], South : 1>
// SPLIT:   aie.packet_dest<%[[VIA]], DMA : 0>
// SPLIT: }

// ROUTED: %[[VIA:.*]] = aie.tile(0, 3)
// ROUTED: aie.switchbox(%[[VIA]]) {
// ROUTED-DAG: aie.masterset(DMA : 0, %[[TO_DMA:.*]])
// ROUTED-DAG: aie.rule(31, 1, %[[TO_DMA]])
// ROUTED-DAG: aie.rule(31, 2, %[[TO_DMA]])
// ROUTED: }
// ROUTED-NOT: aie.masterset(DMA : 0

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
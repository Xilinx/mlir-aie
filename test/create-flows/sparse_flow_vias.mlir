//===- sparse_flow_vias.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-split-flow-vias %s | FileCheck %s --check-prefix=SPLIT
// RUN: aie-opt --split-input-file --aie-split-flow-vias --aie-create-pathfinder-flows %s | FileCheck %s --check-prefix=ROUTED

// A sparse route can have router-filled gaps before, between, and after vias.
// A gap before a directional ingress ends at the adjacent switchbox output.

// SPLIT-LABEL: aie.device(xcvc1902)
// SPLIT-DAG: %[[T01:.*]] = aie.tile(0, 1)
// SPLIT-DAG: %[[T02:.*]] = aie.tile(0, 2)
// SPLIT-DAG: %[[T03:.*]] = aie.tile(0, 3)
// SPLIT-DAG: %[[T04:.*]] = aie.tile(0, 4)
// SPLIT-DAG: %[[T05:.*]] = aie.tile(0, 5)
// SPLIT-DAG: %[[T06:.*]] = aie.tile(0, 6)
// SPLIT-DAG: %[[T07:.*]] = aie.tile(0, 7)
// SPLIT-DAG: %[[T08:.*]] = aie.tile(0, 8)
// SPLIT: aie.flow(%[[T01]], DMA : 0, %[[T02]], North : 0)
// SPLIT: aie.flow(%[[T03]], South : 0, %[[T03]], North : 0)
// SPLIT: aie.flow(%[[T04]], South : 0, %[[T05]], North : 0)
// SPLIT: aie.flow(%[[T06]], South : 0, %[[T06]], North : 0)
// SPLIT: aie.flow(%[[T07]], South : 0, %[[T08]], DMA : 0)

// ROUTED-LABEL: aie.device(xcvc1902)
// ROUTED: aie.switchbox(%{{.*}}) {
// ROUTED:   aie.connect<South : 0, North : 0>
// ROUTED: aie.switchbox(%{{.*}}) {
// ROUTED:   aie.connect<South : 0, North : 4>
// ROUTED: aie.switchbox(%{{.*}}) {
// ROUTED:   aie.connect<South : 4, North : 0>
module {
  aie.device(xcvc1902) {
    %src = aie.tile(0, 1)
    %via0 = aie.tile(0, 3)
    %via1 = aie.tile(0, 6)
    %dst = aie.tile(0, 8)
    aie.flow(%src, DMA : 0, %dst, DMA : 0) via (%via0 : South : 0 -> North : 0, %via1 : South : 0 -> North : 0)
  }
}

// -----

// SPLIT-LABEL: aie.device(xcvc1902)
// SPLIT-DAG: %[[T08:.*]] = aie.tile(0, 8)
// SPLIT-DAG: %[[T07:.*]] = aie.tile(0, 7)
// SPLIT-DAG: %[[T06:.*]] = aie.tile(0, 6)
// SPLIT-DAG: %[[T05:.*]] = aie.tile(0, 5)
// SPLIT-DAG: %[[T04:.*]] = aie.tile(0, 4)
// SPLIT-DAG: %[[T03:.*]] = aie.tile(0, 3)
// SPLIT-DAG: %[[T02:.*]] = aie.tile(0, 2)
// SPLIT-DAG: %[[T01:.*]] = aie.tile(0, 1)
// SPLIT: aie.packet_flow(1) {
// SPLIT:   aie.packet_source<%[[T08]], DMA : 0>
// SPLIT:   aie.packet_dest<%[[T07]], South : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(1) {
// SPLIT:   aie.packet_source<%[[T06]], North : 0>
// SPLIT:   aie.packet_dest<%[[T06]], South : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(1) {
// SPLIT:   aie.packet_source<%[[T05]], North : 0>
// SPLIT:   aie.packet_dest<%[[T04]], South : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(1) {
// SPLIT:   aie.packet_source<%[[T03]], North : 0>
// SPLIT:   aie.packet_dest<%[[T03]], South : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(1) {
// SPLIT:   aie.packet_source<%[[T02]], North : 0>
// SPLIT:   aie.packet_dest<%[[T01]], DMA : 0>
// SPLIT: }

// ROUTED-LABEL: aie.device(xcvc1902)
// ROUTED: aie.packet_rules(DMA : 0)
// ROUTED: aie.packet_rules(North : 0)
module {
  aie.device(xcvc1902) {
    %src = aie.tile(0, 8)
    %via0 = aie.tile(0, 6)
    %via1 = aie.tile(0, 3)
    %dst = aie.tile(0, 1)
    aie.packet_flow(1) {
      aie.packet_source<%src, DMA : 0>
      aie.packet_dest<%dst, DMA : 0>
    } via (%via0 : North : 0 -> South : 0, %via1 : North : 0 -> South : 0)
  }
}

// -----

// SPLIT-LABEL: aie.device(xcvc1902)
// SPLIT-DAG: %[[T12:.*]] = aie.tile(1, 2)
// SPLIT-DAG: %[[T22:.*]] = aie.tile(2, 2)
// SPLIT-DAG: %[[T32:.*]] = aie.tile(3, 2)
// SPLIT-DAG: %[[T42:.*]] = aie.tile(4, 2)
// SPLIT-DAG: %[[T52:.*]] = aie.tile(5, 2)
// SPLIT-DAG: %[[T62:.*]] = aie.tile(6, 2)
// SPLIT-DAG: %[[T72:.*]] = aie.tile(7, 2)
// SPLIT-DAG: %[[T82:.*]] = aie.tile(8, 2)
// SPLIT: aie.packet_flow(2) {
// SPLIT:   aie.packet_source<%[[T12]], DMA : 0>
// SPLIT:   aie.packet_dest<%[[T22]], East : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(2) {
// SPLIT:   aie.packet_source<%[[T32]], West : 0>
// SPLIT:   aie.packet_dest<%[[T32]], East : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(2) {
// SPLIT:   aie.packet_source<%[[T42]], West : 0>
// SPLIT:   aie.packet_dest<%[[T52]], East : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(2) {
// SPLIT:   aie.packet_source<%[[T62]], West : 0>
// SPLIT:   aie.packet_dest<%[[T62]], East : 0>
// SPLIT: }
// SPLIT: aie.packet_flow(2) {
// SPLIT:   aie.packet_source<%[[T72]], West : 0>
// SPLIT:   aie.packet_dest<%[[T82]], DMA : 0>
// SPLIT: }

// ROUTED-LABEL: aie.device(xcvc1902)
// ROUTED: aie.packet_rules(DMA : 0)
// ROUTED: aie.packet_rules(West : 0)
module {
  aie.device(xcvc1902) {
    %src = aie.tile(1, 2)
    %via0 = aie.tile(3, 2)
    %via1 = aie.tile(6, 2)
    %dst = aie.tile(8, 2)
    aie.packet_flow(2) {
      aie.packet_source<%src, DMA : 0>
      aie.packet_dest<%dst, DMA : 0>
    } via (%via0 : West : 0 -> East : 0, %via1 : West : 0 -> East : 0)
  }
}

// -----

// SPLIT-LABEL: aie.device(xcvc1902)
// SPLIT-DAG: %[[T82:.*]] = aie.tile(8, 2)
// SPLIT-DAG: %[[T72:.*]] = aie.tile(7, 2)
// SPLIT-DAG: %[[T52:.*]] = aie.tile(5, 2)
// SPLIT-DAG: %[[T62:.*]] = aie.tile(6, 2)
// SPLIT-DAG: %[[T42:.*]] = aie.tile(4, 2)
// SPLIT-DAG: %[[T22:.*]] = aie.tile(2, 2)
// SPLIT-DAG: %[[T32:.*]] = aie.tile(3, 2)
// SPLIT-DAG: %[[T12:.*]] = aie.tile(1, 2)
// SPLIT: aie.flow(%[[T82]], DMA : 0, %[[T72]], West : 0)
// SPLIT: aie.flow(%[[T62]], East : 0, %[[T62]], West : 0)
// SPLIT: aie.flow(%[[T52]], East : 0, %[[T42]], West : 0)
// SPLIT: aie.flow(%[[T32]], East : 0, %[[T32]], West : 0)
// SPLIT: aie.flow(%[[T22]], East : 0, %[[T12]], DMA : 0)

// ROUTED-LABEL: aie.device(xcvc1902)
// ROUTED: aie.switchbox(%{{.*}}) {
// ROUTED:   aie.connect<East : 0, West : 0>
// ROUTED: aie.switchbox(%{{.*}}) {
// ROUTED:   aie.connect<DMA : 0, West : 0>
// ROUTED: aie.switchbox(%{{.*}}) {
// ROUTED:   aie.connect<East : 0, DMA : 0>
module {
  aie.device(xcvc1902) {
    %src = aie.tile(8, 2)
    %via0 = aie.tile(6, 2)
    %via1 = aie.tile(3, 2)
    %dst = aie.tile(1, 2)
    aie.flow(%src, DMA : 0, %dst, DMA : 0) via (%via0 : East : 0 -> West : 0, %via1 : East : 0 -> West : 0)
  }
}

// -----

// DMA : 0 reaches shim switchbox input South : 3 through the shim mux. The
// trailing gap starts across the via's North wire on the adjacent memory tile.

// SPLIT-LABEL: aie.device(npu2_3col)
// SPLIT-DAG: %[[T10:.*]] = aie.tile(1, 0)
// SPLIT-DAG: %[[T11:.*]] = aie.tile(1, 1)
// SPLIT-DAG: %[[T13:.*]] = aie.tile(1, 3)
// SPLIT: aie.flow(%[[T10]], DMA : 0, %[[T10]], North : 0)
// SPLIT: aie.flow(%[[T11]], South : 0, %[[T13]], DMA : 0)

// ROUTED-LABEL: aie.device(npu2_3col)
// ROUTED: aie.switchbox(%[[RT10:.*]]) {
// ROUTED:   aie.connect<South : 3, North : 0>
// ROUTED: aie.shim_mux(%[[RT10]]) {
// ROUTED:   aie.connect<DMA : 0, North : 3>
module {
  aie.device(npu2_3col) {
    %shim = aie.tile(1, 0)
    %dst = aie.tile(1, 3)
    aie.flow(%shim, DMA : 0, %dst, DMA : 0) via (%shim : South : 3 -> North : 0)
  }
}

//===- flow_vias.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A flow's `via` list pins the tiles and stream-switch ports it routes through.

// RUN: aie-opt %s | aie-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: aie-opt --aie-split-flow-vias %s | FileCheck %s

// ROUNDTRIP: %[[T02:.*]] = aie.tile(0, 2)
// ROUNDTRIP: %[[T03:.*]] = aie.tile(0, 3)
// ROUNDTRIP: aie.flow(%[[T02]], DMA : 0, %[[T03]], DMA : 0) via (%[[T02]] : DMA : 0 -> North : 4, %[[T03]] : South : 4 -> DMA : 0)

// --aie-split-flow-vias rewrites each pinned switchbox hop into a local flow.
// The router assigns the switchbox resources for these local flows together
// with the gaps between them.
// CHECK: %[[T02:.*]] = aie.tile(0, 2)
// CHECK: %[[T03:.*]] = aie.tile(0, 3)
// CHECK: %[[T05:.*]] = aie.tile(0, 5)
// CHECK: aie.flow(%[[T02]], DMA : 0, %[[T02]], North : 4)
// CHECK: aie.flow(%[[T03]], South : 4, %[[T03]], DMA : 0)
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%[[T02]], DMA : 1>
// CHECK:   aie.packet_dest<%[[T03]], South : 0>
// CHECK: } {priority_route = true}
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%[[T03]], South : 0>
// CHECK:   aie.packet_dest<%[[T03]], North : 0>
// CHECK: } {priority_route = true}
// CHECK: aie.packet_flow(1) {
// CHECK:   aie.packet_source<%[[T03]], North : 0>
// CHECK:   aie.packet_dest<%[[T05]], DMA : 0>
// CHECK: } {keep_pkt_header = true, priority_route = true}
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t05 = aie.tile(0, 5)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0) via (%t02 : DMA : 0 -> North : 4, %t03 : South : 4 -> DMA : 0)
    aie.packet_flow(1) {
      aie.packet_source<%t02, DMA : 1>
      aie.packet_dest<%t05, DMA : 0>
    } via (%t03 : South : 0 -> North : 0) {keep_pkt_header = true, priority_route = true}
  }
}

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

// --aie-split-flow-vias rewrites the via flow into a pinned switchbox at each
// via tile plus the segment flows between the pinned ports.  Segments that
// coincide with the flow's own source/dest port (here both ends) are elided,
// leaving just the inter-switchbox hop.
// CHECK: %[[T02:.*]] = aie.tile(0, 2)
// CHECK: aie.switchbox(%[[T02]]) {
// CHECK:   aie.connect<DMA : 0, North : 4>
// CHECK: }
// CHECK: %[[T03:.*]] = aie.tile(0, 3)
// CHECK: aie.switchbox(%[[T03]]) {
// CHECK:   aie.connect<South : 4, DMA : 0>
// CHECK: }
// CHECK: aie.flow(%[[T02]], North : 4, %[[T03]], South : 4)
module {
  aie.device(xcvc1902) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.flow(%t02, DMA : 0, %t03, DMA : 0) via (%t02 : DMA : 0 -> North : 4, %t03 : South : 4 -> DMA : 0)
  }
}

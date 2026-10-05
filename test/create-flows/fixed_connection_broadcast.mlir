//===- fixed_connection_broadcast.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for ingesting a pre-lowered broadcast switchbox alongside a
// freshly added flow.  addFixedConnection() must accept a broadcast -- one
// source port fanning out to several destination ports -- rather than rejecting
// the sibling connects.
//
// Setup
// -----
//   tile(2,2) already contains a lowered broadcast: source North:0 drives both
//   South:0 and East:0.  (This is what --aie-create-pathfinder-flows emits for
//   several aie.flows sharing one source.)
//
//   An independent flow tile(2,3) DMA:0 -> tile(2,1) DMA:1 is added at the
//   physical level, without lifting the switchbox back to flows.
//
// Before the fix addFixedConnection() invalidated the whole North:0 source row
// when it recorded the first fan-out connect, so the second (North:0 -> East:0)
// failed its AVAILABLE check and the pass aborted with
// "Unable to add fixed connections".  The broadcast is now ingested and the new
// flow routes around the reserved ports.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK: %[[T22:.*]] = aie.tile(2, 2)

// The broadcast is preserved intact: both fan-out connects survive.
// CHECK:      aie.switchbox(%[[T22]]) {
// CHECK-DAG:    aie.connect<North : 0, South : 0>
// CHECK-DAG:    aie.connect<North : 0, East : 0>
// CHECK:      }

module {
  aie.device(xcvc1902) {
    %t21 = aie.tile(2, 1)
    %t22 = aie.tile(2, 2)
    %t23 = aie.tile(2, 3)
    %t32 = aie.tile(3, 2)

    // Pre-lowered broadcast at tile(2,2): North:0 -> {South:0, East:0}.
    %sb22 = aie.switchbox(%t22) {
      aie.connect<North : 0, South : 0>
      aie.connect<North : 0, East : 0>
    }

    // New flow added alongside the fixed broadcast switchbox.
    aie.flow(%t23, DMA : 0, %t21, DMA : 1)
  }
}

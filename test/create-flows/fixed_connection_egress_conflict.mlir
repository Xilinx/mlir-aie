//===- fixed_connection_egress_conflict.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for the egress side of addFixedConnection(): a circuit
// ConnectOp monopolizes its destination (output) port, not just its source
// port.  addFixedConnection() must mark all connectivity[*][dst] cells INVALID
// so the pathfinder cannot drive a second stream onto an occupied output.
//
// Setup
// -----
//   tile(2,2) has a pre-existing circuit connection: North:1 -> South:0.
//   This occupies output port South:0 (and the wire to tile(2,1)).
//
//   A flow is requested from tile(2,3) DMA:0 to tile(2,1) DMA:0.  The
//   straight-line path goes down column 2 through tile(2,2), entering from the
//   North and exiting to the South.
//
// Without the egress reservation the pathfinder is free to route the flow
// through tile(2,2) as North:0 -> South:0, colliding with the pre-existing
// ConnectOp on output South:0; the verifier then rejects the switchbox
// ("different destinations") and aie-opt exits non-zero.
//
// With the egress reservation output South:0 is unavailable, so the flow exits
// tile(2,2) on a different south channel and the pre-existing connect is
// preserved untouched.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK: %[[T22:.*]] = aie.tile(2, 2)

// tile(2,2) keeps its original output connection and gains no second connect
// driving South:0.
// CHECK:      aie.switchbox(%[[T22]]) {
// CHECK:        aie.connect<North : 1, South : 0>
// CHECK-NOT:    South : 0>
// CHECK:      }

module {
  aie.device(xcvc1902) {
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    // Fixed circuit connection at tile(2,2): occupies output port South:0.
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<North : 1, South : 0>
    }

    // Straight-line flow down column 2 through tile(2,2).  Must not reuse the
    // occupied South:0 output port.
    aie.flow(%tile_2_3, DMA : 0, %tile_2_1, DMA : 0)
  }
}

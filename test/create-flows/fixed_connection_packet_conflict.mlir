//===- fixed_connection_packet_conflict.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for: addFixedConnection() only invalidates specific (src,
// dst) pair, not the full source port (issue #2931).
//
// A circuit-switched ConnectOp monopolizes its entire source port -- no
// PacketRulesOp may share it.  addFixedConnection() must therefore mark all
// connectivity[src][*] cells INVALID when it records a ConnectOp, not just
// the single connectivity[src][dst] cell.
//
// Setup
// -----
//   tile(2,2) has a pre-existing circuit connection: North:0 -> DMA:0.
//   This locks source port North:0 at tile(2,2).
//
//   A packet_flow is requested from tile(2,3) DMA:0 to tile(2,1) DMA:0.
//   The straight-line path goes through tile(2,2) entering on North:0.
//
// Bug behaviour (before fix)
// --------------------------
//   addFixedConnection marks only connectivity[North:0][DMA:0] INVALID.
//   connectivity[North:0][South:0] remains AVAILABLE, so the pathfinder
//   routes the packet flow through North:0 -> South:0 at tile(2,2).
//   The MLIR verifier rejects this IR:
//     "packet switched source North0 cannot match another connect or
//      masterset operation"
//   aie-opt exits non-zero and this test FAILS.
//
// Fixed behaviour
// ---------------
//   addFixedConnection marks all connectivity[North:0][*] INVALID.
//   The straight-line path is blocked; the pathfinder routes via column 1:
//     tile(2,3) -> West -> tile(1,3) -> South -> tile(1,2) ->
//     South -> tile(1,1) -> East -> tile(2,1)
//   tile(2,2) retains only the original circuit connect<North:0, DMA:0>
//   and no packet_rules using North:0.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// After the fix the packet flow is routed via column 1.
// tile(2,2) must contain only the original circuit ConnectOp and must NOT
// have any packet_rules on North:0.

// CHECK: %[[T11:.*]] = aie.tile(1, 1)
// CHECK: %[[T12:.*]] = aie.tile(1, 2)
// CHECK: %[[T13:.*]] = aie.tile(1, 3)
// CHECK: %[[T21:.*]] = aie.tile(2, 1)
// CHECK: %[[T22:.*]] = aie.tile(2, 2)
// CHECK: %[[T23:.*]] = aie.tile(2, 3)

// The fixed switchbox at (2,2) keeps the original circuit ConnectOp and
// must not gain a packet_rules on North:0.
// CHECK:      %switchbox_2_2 = aie.switchbox(%[[T22]]) {
// CHECK-NEXT:   aie.connect<North : 0, DMA : 0>
// CHECK-NOT:    aie.packet_rules(North : 0)
// CHECK:      }

module {
  aie.device(xcvc1902) {
    // Tiles in column 1 provide the alternate route after the fix.
    %tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    // Fixed circuit connection at tile(2,2): North:0 -> DMA:0.
    // Monopolizes source port North:0; no packet flow may also use it.
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<North : 0, DMA : 0>
    }

    // Packet flow: tile(2,3) DMA:0 -> tile(2,1) DMA:0.
    // Without the fix: pathfinder routes through tile(2,2) North:0->South:0,
    // conflicting with the circuit ConnectOp above; verifier rejects.
    // With the fix: pathfinder uses the column-1 alternate route.
    aie.packet_flow(0x0) {
      aie.packet_source<%tile_2_3, DMA : 0>
      aie.packet_dest<%tile_2_1, DMA : 0>
    }
  }
}

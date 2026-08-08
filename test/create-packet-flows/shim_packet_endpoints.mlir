//===- shim_packet_endpoints.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A packet endpoint on a shim DMA is not a switchbox port: the router rewrites
// it to the shim's South port and adds the matching aie.shim_mux connection.
// Each shim DMA channel has its own fixed mux wiring, so exercise all three
// here -- MM2S 1, S2MM 0 and S2MM 1. MM2S 0 is covered by find_flows.mlir.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: aie.shim_mux(%shim_noc_tile_0_0)
// CHECK-DAG:     aie.connect<DMA : 1, North : 7>
// CHECK-DAG:     aie.connect<North : 2, DMA : 0>
// CHECK-DAG:     aie.connect<North : 3, DMA : 1>
module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)

    // Source on shim MM2S 1.
    aie.packet_flow(5) { aie.packet_source<%t00, DMA : 1>  aie.packet_dest<%t01, DMA : 0> }
    // Destinations on shim S2MM 0 and S2MM 1.
    aie.packet_flow(9) { aie.packet_source<%t02, DMA : 0>  aie.packet_dest<%t00, DMA : 0> }
    aie.packet_flow(11) { aie.packet_source<%t02, DMA : 1>  aie.packet_dest<%t00, DMA : 1> }
  }
}

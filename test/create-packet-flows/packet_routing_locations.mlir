//===- packet_routing_locations.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The switchbox settings a packet flow becomes carry the flow's location, so a
// rule or master set names the flows behind it rather than its tile.

// RUN: aie-opt --aie-create-pathfinder-flows --mlir-print-debuginfo --mlir-print-local-scope %s | FileCheck %s

// The shim mux connection into the switchbox carries them too.
// CHECK-LABEL: aie.shim_mux(%shim_noc_tile_0_0)
// CHECK: aie.connect<DMA : 0, North : {{[0-9]+}}> loc(fused["iron:act_in"("design.py":10:0), "iron:wts_in"("design.py":11:0)])

// Both flows enter the shim's switchbox by one slave port and leave it by one
// master port, so those settings name both, in the order they were written.
// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK-DAG: aie.masterset(North : {{[0-9]+}}, %{{[0-9]+}}) loc(fused["iron:act_in"("design.py":10:0), "iron:wts_in"("design.py":11:0)])
// CHECK-DAG: aie.rule({{[0-9]+}}, {{[0-9]+}}, %{{[0-9]+}}) loc(fused["iron:act_in"("design.py":10:0), "iron:wts_in"("design.py":11:0)])
// CHECK-DAG: } loc(fused["iron:act_in"("design.py":10:0), "iron:wts_in"("design.py":11:0)])
// Settings no flow asked for keep the tile's location.
// CHECK-DAG: aie.amsel<0> (0) loc("tile.py":1:0)
// CHECK: } loc("tile.py":1:0)

// At the far end the flows part ways.
// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK-DAG: aie.masterset(DMA : 0, %{{[0-9]+}}) loc("iron:act_in"("design.py":10:0))
// CHECK-DAG: aie.masterset(DMA : 1, %{{[0-9]+}}) loc("iron:wts_in"("design.py":11:0))
// Their rules still share the slave port's one packet_rules, which names both.
// CHECK-DAG: aie.rule({{[0-9]+}}, 1, %{{[0-9]+}}) loc("iron:act_in"("design.py":10:0))
// CHECK-DAG: aie.rule({{[0-9]+}}, 2, %{{[0-9]+}}) loc("iron:wts_in"("design.py":11:0))
// CHECK-DAG: } loc(fused["iron:act_in"("design.py":10:0), "iron:wts_in"("design.py":11:0)])

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0) loc("tile.py":1:0)
    %tile_0_2 = aie.tile(0, 2) loc("tile.py":2:0)
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    } loc("iron:act_in"("design.py":10:0))
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 1>
    } loc("iron:wts_in"("design.py":11:0))
  }
}

//===- find_flows_shim_mux_vestigial.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A shim-mux connection is not a switchbox transit, so it cannot be pinned by a
// via. A shim-mux connection whose stream never enters a switchbox (a dead
// input mux like DMA:1 -> North:7 here) therefore has nothing to lift into; it
// must be left materialized rather than turned into a via-less flow that
// --aie-create-pathfinder-flows could not route. The live shim-mux connection
// feeding the packet route (DMA:0 -> North:3) is instead reconstructed by
// --aie-split-flow-vias, so the whole design still round-trips.

// RUN: aie-opt --aie-find-flows=emit-vias=true %s | FileCheck %s
// RUN: aie-opt --aie-find-flows=emit-vias=true %s | aie-opt --aie-split-flow-vias | aie-opt --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUNDTRIP

// The dead input mux stays materialized, and nothing lifts it into a flow.
// CHECK: aie.shim_mux(%{{.*}}) {
// CHECK:   aie.connect<DMA : 1, North : 7>
// CHECK: }
// CHECK-NOT: aie.flow(
// The live packet route lifts with its via chain.
// CHECK: aie.packet_flow(0) {
// CHECK:   aie.packet_source<%{{.*}}, DMA : 0>
// CHECK:   aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK: } via (

// Re-lowering keeps the dead mux and regenerates the live one, and routes.
// ROUNDTRIP: aie.shim_mux(%{{.*}}) {
// ROUNDTRIP-DAG: aie.connect<DMA : 1, North : 7>
// ROUNDTRIP-DAG: aie.connect<DMA : 0, North : 3>
// ROUNDTRIP: }

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %mem_tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 1, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 1) {
        aie.rule(31, 0, %0)
      }
    }
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
  }
}

//===- find_flows_split_at_attr.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A lifted flow is rebuilt from its via chain plus the attributes carried on the
// flow, so an attribute it cannot reconstruct -- here keep_pkt_header on an
// intermediate master set -- would otherwise be dropped. Instead the flow is
// split at that switchbox: the attributed switchbox (0, 3) stays materialized
// verbatim while the sections before and after it lift, so the round-trip
// preserves the attribute rather than silently dropping it.

// RUN: aie-opt --aie-find-flows=emit-vias=true %s | FileCheck %s
// RUN: aie-opt --aie-find-flows=emit-vias=true %s | aie-opt --aie-split-flow-vias | aie-opt --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUNDTRIP

// The attributed hop stays materialized verbatim.
// CHECK: %[[T03:.*]] = aie.tile(0, 3)
// CHECK: aie.switchbox(%[[T03]]) {
// CHECK:   aie.masterset(North : 0, %{{.*}}) {keep_pkt_header = true}
// CHECK:   aie.packet_rules(South : 3) {
// CHECK:     aie.rule(31, 0, %{{.*}})
// CHECK:   }
// CHECK: }
// The flow lifts on both sides of the attributed switchbox.
// CHECK: aie.packet_flow(0 mask 31) {
// CHECK:   aie.packet_source<%{{.*}}, DMA : 0>
// CHECK:   aie.packet_dest<%[[T03]], South : 3>
// CHECK: } via (
// CHECK: aie.packet_flow(0 mask 31) {
// CHECK:   aie.packet_source<%[[T03]], North : 0>
// CHECK:   aie.packet_dest<%{{.*}}, DMA : 0>
// CHECK: } via (

// After splitting the vias and re-routing, keep_pkt_header still rides on the
// intermediate master set.
// ROUNDTRIP: aie.masterset(North : 0, %{{.*}}) {keep_pkt_header = true}

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 3, %0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_3 = aie.tile(0, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 3) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_4 = aie.tile(0, 4)
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(North : 4, %0)
      aie.packet_rules(South : 0) {
        aie.rule(31, 0, %0)
      }
    }
    %tile_0_5 = aie.tile(0, 5)
    %switchbox_0_5 = aie.switchbox(%tile_0_5) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(DMA : 0, %0)
      aie.packet_rules(South : 4) {
        aie.rule(31, 0, %0)
      }
    }
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%switchbox_0_3 : North, %switchbox_0_4 : South)
    aie.wire(%switchbox_0_4 : North, %switchbox_0_5 : South)
    aie.wire(%tile_0_5 : DMA, %switchbox_0_5 : DMA)
  }
}

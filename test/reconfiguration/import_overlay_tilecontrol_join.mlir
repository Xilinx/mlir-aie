// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: split-file %s %t
// RUN: aie-opt --aie-import-overlay --aie-create-pathfinder-flows --split-input-file %t/reuse.mlir | FileCheck %s
// RUN: aie-opt --aie-import-overlay --aie-create-pathfinder-flows --split-input-file --verify-diagnostics %t/incompatible.mlir -o /dev/null

//--- reuse.mlir

// CHECK-LABEL: aie.device(npu2_1col) @application
// CHECK: %[[A:.*]] = aie.amsel<5> (3) {aie.is_ctrl_pkt_overlay}
// CHECK-NOT: aie.masterset(TileControl
// CHECK: aie.masterset(TileControl : 0, %[[A]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NOT: aie.masterset(TileControl
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 0, %[[A]])
// CHECK-LABEL: aie.device(npu2_1col) @transport

module {
  aie.device(npu2_1col) @application {
    %tile = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    }
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<5>(3) {aie.is_ctrl_pkt_overlay}
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %a) {aie.is_ctrl_pkt_overlay}
      }
    }
  } {aie.is_ctrl_pkt_overlay}
}

// -----

// CHECK-LABEL: aie.device(npu2_1col) @compatible
// CHECK: %[[SHARED:.*]] = aie.amsel<5> (3) {aie.is_ctrl_pkt_overlay}
// CHECK: %[[EXCLUSIVE:.*]] = aie.amsel<5> (2) {aie.is_ctrl_pkt_overlay}
// CHECK: aie.masterset(TileControl : 0, %[[SHARED]], %[[EXCLUSIVE]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NEXT: {{.*}}aie.masterset(North : 0, %[[SHARED]]) {aie.is_ctrl_pkt_overlay}
// CHECK: aie.rule(31, 15, %[[SHARED]]) {aie.is_ctrl_pkt_overlay}
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 0, %[[EXCLUSIVE]])
// CHECK-LABEL: aie.device(npu2_1col) @transport
module {
  aie.device(npu2_1col) @compatible {
    %tile = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    } {keep_pkt_header = true}
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    %neighbor = aie.tile(0, 3)
    aie.switchbox(%tile) {
      %shared = aie.amsel<5>(3) {aie.is_ctrl_pkt_overlay}
      %exclusive = aie.amsel<5>(2) {aie.is_ctrl_pkt_overlay}
      aie.masterset(TileControl : 0, %shared, %exclusive) {aie.is_ctrl_pkt_overlay}
      aie.masterset(North : 0, %shared) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %shared) {aie.is_ctrl_pkt_overlay}
      }
    }
    aie.switchbox(%neighbor) {
      %receive = aie.amsel<5>(3) {aie.is_ctrl_pkt_overlay}
      aie.masterset(TileControl : 0, %receive) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(South : 0) {
        aie.rule(0, 0, %receive) {aie.is_ctrl_pkt_overlay}
      }
    }
  } {aie.is_ctrl_pkt_overlay}
}

//--- incompatible.mlir
module {
  aie.device(npu2_1col) @application {
    %tile = aie.tile(0, 2)
    // expected-error@+1 {{control packet overlay has no TileControl-only amsel}}
    aie.packet_flow(0) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    } {keep_pkt_header = true}
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %shared = aie.amsel<5>(3) {aie.is_ctrl_pkt_overlay}
      // expected-note@+1 {{TileControl master set declared here}}
      aie.masterset(TileControl : 0, %shared) {aie.is_ctrl_pkt_overlay}
      aie.masterset(North : 0, %shared) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %shared) {aie.is_ctrl_pkt_overlay}
      }
    }
  } {aie.is_ctrl_pkt_overlay}
}

// -----

module {
  aie.device(npu2_1col) @hardware_alias {
    %tile = aie.tile(0, 2)
    // expected-error@+1 {{control packet overlay has no TileControl-only amsel}}
    aie.packet_flow(0) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    } {keep_pkt_header = true}
    aie.switchbox(%tile) {
      %local = aie.amsel<5>(3) {aie.is_ctrl_pkt_overlay}
      %north = aie.amsel<5>(3) {aie.is_ctrl_pkt_overlay}
      // expected-note@+1 {{TileControl master set declared here}}
      aie.masterset(TileControl : 0, %local) {aie.is_ctrl_pkt_overlay}
      aie.masterset(North : 0, %north) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %local) {aie.is_ctrl_pkt_overlay}
      }
    }
  }
}

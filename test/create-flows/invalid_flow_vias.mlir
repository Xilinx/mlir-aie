//===- invalid_flow_vias.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-create-pathfinder-flows %s

module {
  aie.device(npu1_1col) {
    %src = aie.tile(0, 2)
    %via = aie.tile(0, 3)
    %dst0 = aie.tile(0, 4)
    %dst1 = aie.tile(0, 5)
    // expected-error@+1 {{with via waypoints must have exactly one aie.packet_dest}}
    aie.packet_flow(1) {
      aie.packet_source<%src, DMA : 0>
      aie.packet_dest<%dst0, DMA : 0>
      aie.packet_dest<%dst1, DMA : 0>
    } via (%via : South : 0 -> North : 0)
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %src = aie.tile(0, 2)
    %via = aie.tile(0, 3)
    %dst0 = aie.tile(0, 4)
    %dst1 = aie.tile(0, 5)
    // expected-note@+1 {{via 0 claims the same circuit egress}}
    aie.flow(%src, DMA : 0, %dst0, DMA : 0) via (%via : South : 0 -> North : 0)
    // expected-error@+1 {{via 0 claims circuit egress (0, 3) North : 0, which another via already claims}}
    aie.flow(%src, DMA : 0, %dst1, DMA : 0) via (%via : South : 0 -> North : 0)
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %src0 = aie.tile(0, 2)
    %src1 = aie.tile(0, 3)
    %via = aie.tile(0, 4)
    %dst = aie.tile(0, 5)
    // expected-error@+1 {{with via waypoints must have exactly one aie.packet_source}}
    aie.packet_flow(1) {
      aie.packet_source<%src0, DMA : 0>
      aie.packet_source<%src1, DMA : 0>
      aie.packet_dest<%dst, DMA : 0>
    } via (%via : South : 0 -> North : 0)
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %src = aie.tile(0, 2)
    %via = aie.tile(0, 3)
    %dst = aie.tile(0, 4)
    // expected-error@+1 {{has negative via ingress channel -1 at index 0}}
    aie.flow(%src, DMA : 0, %dst, DMA : 0) via (%via : South : -1 -> North : 0)
  }
}

// -----

"builtin.module"() ({
  "aie.device"() <{device = 5 : i32, sym_name = "main"}> ({
    %src = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index
    %via = "aie.tile"() <{col = 0 : i32, row = 3 : i32}> : () -> index
    %dst = "aie.tile"() <{col = 0 : i32, row = 4 : i32}> : () -> index
    // expected-error@+1 {{has invalid via ingress bundle 99 at index 0}}
    "aie.flow"(%src, %dst, %via) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32, via_egress_bundles = array<i32: 5>, via_egress_channels = array<i32: 0>, via_ingress_bundles = array<i32: 99>, via_ingress_channels = array<i32: 0>}> : (index, index, index) -> ()
    "aie.end"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

"builtin.module"() ({
  "aie.device"() <{device = 5 : i32, sym_name = "main"}> ({
    %src = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index
    %via = "aie.tile"() <{col = 0 : i32, row = 3 : i32}> : () -> index
    %dst = "aie.tile"() <{col = 0 : i32, row = 4 : i32}> : () -> index
    // expected-error@+1 {{has negative via egress channel -1 at index 0}}
    "aie.packet_flow"(%via) <{ID = 1 : i8, via_egress_bundles = array<i32: 5>, via_egress_channels = array<i32: -1>, via_ingress_bundles = array<i32: 3>, via_ingress_channels = array<i32: 0>}> ({
      "aie.packet_source"(%src) <{bundle = 1 : i32, channel = 0 : i32}> : (index) -> ()
      "aie.packet_dest"(%dst) <{bundle = 1 : i32, channel = 0 : i32}> : (index) -> ()
      "aie.end"() : () -> ()
    }) : (index) -> ()
    "aie.end"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

"builtin.module"() ({
  "aie.device"() <{device = 5 : i32, sym_name = "main"}> ({
    %src = "aie.tile"() <{col = 0 : i32, row = 2 : i32}> : () -> index
    %via = "arith.constant"() <{value = 0 : index}> : () -> index
    %dst = "aie.tile"() <{col = 0 : i32, row = 4 : i32}> : () -> index
    // expected-error@+1 {{has via operand at index 0 that is not defined by a tile-like operation}}
    "aie.flow"(%src, %dst, %via) <{dest_bundle = 1 : i32, dest_channel = 0 : i32, source_bundle = 1 : i32, source_channel = 0 : i32, via_egress_bundles = array<i32: 5>, via_egress_channels = array<i32: 0>, via_ingress_bundles = array<i32: 3>, via_ingress_channels = array<i32: 0>}> : (index, index, index) -> ()
    "aie.end"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----

module {
  aie.device(npu1_1col) {
    %src = aie.tile(0, 2)
    %via = aie.tile(0, 3)
    %dst = aie.tile(0, 4)
    // expected-error@+1 {{cannot be routed while it carries via waypoints; run --aie-split-flow-vias first}}
    aie.packet_flow(1) {
      aie.packet_source<%src, DMA : 0>
      aie.packet_dest<%dst, DMA : 0>
    } via (%via : South : 0 -> North : 0)
  }
}
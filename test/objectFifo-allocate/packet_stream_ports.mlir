// RUN: aie-opt --split-input-file --aie-assign-packet-ids --aie-objectfifo-allocate --verify-diagnostics %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Both packet flows keep the one output and the one input port they share.
// CHECK-LABEL: module @shared_packet_port
// CHECK:       aie.packet_flow(0) {
// CHECK-NEXT:    aie.packet_source<%[[S:.*]], Core : 0>
// CHECK-NEXT:    aie.packet_dest<%[[D:.*]], Core : 0>
// CHECK:       aie.packet_flow(1) {
// CHECK-NEXT:    aie.packet_source<%[[S]], Core : 0>
// CHECK-NEXT:    aie.packet_dest<%[[D]], Core : 0>
module @shared_packet_port {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    aie.route_endpoint @s0(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @s1(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d0(%dest) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d1(%dest) Core {channelIndex = 0 : i32}
    aie.route from @s0 to [@d0] {packet = #aie.packet_info<>}
    aie.route from @s1 to [@d1] {packet = #aie.packet_info<>}
  }
}

// -----

// CHECK-LABEL: module @existing_packet_port
// CHECK-COUNT-2: aie.packet_source<%{{.*}}, Core : 0>
module @existing_packet_port {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    aie.packet_flow(0x01) {
      aie.packet_source<%source, "Core" : 0>
      aie.packet_dest<%dest, "Core" : 0>
    }
    aie.route_endpoint @s(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d(%dest) Core {channelIndex = 0 : i32}
    aie.route from @s to [@d] {packet = #aie.packet_info<>}
  }
}

// -----

module @packet_against_circuit {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    // expected-note @+1 {{the other stream is here}}
    aie.route_endpoint @packet_source(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @packet_dest(%dest) Core {channelIndex = 0 : i32}
    %alias = aie.logical_tile<CoreTile>(0, 2)
    // expected-error @+1 {{Core output 0 is already in use on this tile}}
    aie.route_endpoint @circuit_source(%alias) Core {channelIndex = 0 : i32}
    aie.route_endpoint @circuit_dest(%dest) Core {channelIndex = 1 : i32}
    aie.route from @packet_source to [@packet_dest] {packet = #aie.packet_info<>}
    aie.route from @circuit_source to [@circuit_dest]
  }
}

// -----

module @existing_packet_against_circuit {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    aie.packet_flow(0x01) {
      // expected-note @+1 {{the other stream is here}}
      aie.packet_source<%source, "Core" : 0>
      aie.packet_dest<%dest, "Core" : 0>
    }
    %alias = aie.logical_tile<CoreTile>(0, 2)
    // expected-error @+1 {{Core output 0 is already in use on this tile}}
    aie.route_endpoint @s(%alias) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d(%dest) Core {channelIndex = 1 : i32}
    aie.route from @s to [@d]
  }
}

// -----

module @packet_against_existing_circuit {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    // expected-note @+1 {{the other stream is here}}
    aie.flow(%source, Core : 0, %dest, Core : 0)
    %alias = aie.logical_tile<CoreTile>(0, 2)
    // expected-error @+1 {{Core output 0 is already in use on this tile}}
    aie.route_endpoint @s(%alias) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d(%dest) Core {channelIndex = 1 : i32}
    aie.route from @s to [@d] {packet = #aie.packet_info<>}
  }
}

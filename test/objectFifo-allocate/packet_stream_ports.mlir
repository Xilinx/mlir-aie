// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --verify-diagnostics %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module @shared_packet_port {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    aie.route_endpoint @s0(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @s1(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d0(%dest) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d1(%dest) Core {channelIndex = 0 : i32}
    aie.route from @s0 to [@d0] {packet}
    aie.route from @s1 to [@d1] {packet}
  }
}

// -----

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
    aie.route from @s to [@d] {packet}
  }
}

// -----

module @packet_against_circuit {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    aie.route_endpoint @packet_source(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @packet_dest(%dest) Core {channelIndex = 0 : i32}
    // expected-error @+1 {{number of output Core channels exceeded}}
    %alias = aie.logical_tile<CoreTile>(0, 2)
    aie.route_endpoint @circuit_source(%alias) Core {channelIndex = 0 : i32}
    aie.route_endpoint @circuit_dest(%dest) Core {channelIndex = 1 : i32}
    aie.route from @packet_source to [@packet_dest] {packet}
    aie.route from @circuit_source to [@circuit_dest]
  }
}

// -----

module @existing_packet_against_circuit {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    aie.packet_flow(0x01) {
      aie.packet_source<%source, "Core" : 0>
      aie.packet_dest<%dest, "Core" : 0>
    }
    // expected-error @+1 {{number of output Core channels exceeded}}
    %alias = aie.logical_tile<CoreTile>(0, 2)
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
    aie.flow(%source, Core : 0, %dest, Core : 0)
    // expected-error @+1 {{number of output Core channels exceeded}}
    %alias = aie.logical_tile<CoreTile>(0, 2)
    aie.route_endpoint @s(%alias) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d(%dest) Core {channelIndex = 1 : i32}
    aie.route from @s to [@d] {packet}
  }
}

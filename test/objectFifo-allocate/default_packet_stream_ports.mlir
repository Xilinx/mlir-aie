// RUN: aie-opt --aie-objectfifo-allocate="packet-sw-objFifos=true" %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module @default_packet_routes {
  aie.device(npu2) {
    %source = aie.tile(0, 2)
    %dest = aie.tile(1, 2)
    aie.route_endpoint @s0(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @s1(%source) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d0(%dest) Core {channelIndex = 0 : i32}
    aie.route_endpoint @d1(%dest) Core {channelIndex = 0 : i32}
    aie.route from @s0 to [@d0]
    aie.route from @s1 to [@d1]
  }
}

// CHECK-COUNT-2: aie.packet_flow

//===- assign.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-packet-ids %s | FileCheck %s
// RUN: aie-opt --aie-assign-packet-ids="packet-sw-objFifos=true" %s | FileCheck %s --check-prefix=ALL

// An open header takes the lowest id nothing on the device holds: here 0 is a
// hand-written packet flow, 1 a trace packet, 2 a pinned route, so the open
// route gets 3 and keeps its type. A pinned route is left as written and a
// circuit route stays one, unless the flag makes every route packet-switched.

// CHECK-LABEL: @assign
// CHECK:       aie.route from @a to [@b] {packet = #aie.packet_info<pkt_type = 1, pkt_id = 3>}
// CHECK:       aie.route from @c to [@d] {packet = #aie.packet_info<pkt_id = 2>}
// CHECK:       aie.route from @e to [@f]
// CHECK-NOT:   packet

// ALL-LABEL: @assign
// ALL:       aie.route from @a to [@b] {packet = #aie.packet_info<pkt_type = 1, pkt_id = 3>}
// ALL:       aie.route from @c to [@d] {packet = #aie.packet_info<pkt_id = 2>}
// ALL:       aie.route from @e to [@f] {packet = #aie.packet_info<pkt_id = 4>}
module @assign {
  aie.device(xcve2302) {
    %shim = aie.tile(0, 0)
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.packet_flow(0) {
      aie.packet_source<%tile22, Core : 0>
      aie.packet_dest<%tile23, Core : 0>
    }
    aie.trace @t(%tile22) {
      aie.trace.packet id=1 type=core
    }

    aie.route_endpoint @a(%shim) DMA
    aie.route_endpoint @b(%tile12) Core
    aie.route_endpoint @c(%shim) DMA
    aie.route_endpoint @d(%tile13) Core
    aie.route_endpoint @e(%shim) DMA
    aie.route_endpoint @f(%tile23) Core

    aie.route from @a to [@b] {packet = #aie.packet_info<pkt_type = 1>}
    aie.route from @c to [@d] {packet = #aie.packet_info<pkt_id = 2>}
    aie.route from @e to [@f]
  }
}

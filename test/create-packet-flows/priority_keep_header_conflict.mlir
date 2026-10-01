//===- priority_keep_header_conflict.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-create-pathfinder-flows %s

// The last flow to end at a port sets whether packets keep their header
// there, but a control-packet reload keeps the prioritized flows' setting, so
// the two must agree.

module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t01, DMA : 0> aie.packet_dest<%t02, DMA : 0> } {priority_route = true}
    // expected-error@+1 {{packet flow 2 is the last to end at (0, 2) DMA:0, so it sets whether packets keep their header there, but the prioritized flows (the control overlay) ending there set it otherwise}}
    aie.packet_flow(2) { aie.packet_source<%t03, DMA : 0> aie.packet_dest<%t02, DMA : 0> } {keep_pkt_header = true}
  }
}

// -----

// They agree when the last flow states the default.

module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t01, DMA : 0> aie.packet_dest<%t02, DMA : 0> } {priority_route = true}
    aie.packet_flow(2) { aie.packet_source<%t03, DMA : 0> aie.packet_dest<%t02, DMA : 0> } {keep_pkt_header = false}
  }
}

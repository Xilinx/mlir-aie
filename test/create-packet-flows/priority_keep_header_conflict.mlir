// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: aie-opt --aie-create-pathfinder-flows %s -o /dev/null



// Without a reload, the last flow sets it.

module {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.packet_flow(1) { aie.packet_source<%t01, DMA : 0> aie.packet_dest<%t02, DMA : 0> } {priority_route = true}
    aie.packet_flow(2) { aie.packet_source<%t03, DMA : 0> aie.packet_dest<%t02, DMA : 0> } {keep_pkt_header = true}
  }
}

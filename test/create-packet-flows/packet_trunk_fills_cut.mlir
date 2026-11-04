//===- packet_trunk_fills_cut.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Ten circuit flows and packets 3 and 13 fill the twelve channels from row 3
// to row 2. The id 28 trees from (2, 1) DMA:3 and (2, 2) DMA:1 share two
// receivers, so they meet; (2, 2) DMA:1's id 14 conflicts with (2, 1) DMA:3,
// so it leaves as a part of its own and the trees meet at (2, 1), below the
// boundary. Trunking id 28 north to meet and coming back down takes a
// thirteenth channel.
// Reduced from router_mutation.py npu1 seed 207460330 (port-cap).

module {
  aie.device(npu1_3col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %t_2_0 = aie.tile(2, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_3 = aie.tile(2, 3)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    aie.flow(%t_0_3, Core : 0, %t_1_0, DMA : 1)
    aie.flow(%t_1_4, DMA : 0, %t_0_1, DMA : 0)
    aie.flow(%t_2_4, DMA : 1, %t_1_2, DMA : 1)
    aie.flow(%t_1_3, DMA : 0, %t_1_2, Core : 0)
    aie.flow(%t_1_5, DMA : 1, %t_0_0, DMA : 0)
    aie.flow(%t_0_5, Core : 0, %t_0_2, DMA : 1)
    aie.flow(%t_1_4, DMA : 1, %t_2_1, DMA : 1)
    aie.flow(%t_1_5, DMA : 0, %t_1_1, DMA : 2)
    aie.flow(%t_2_5, DMA : 1, %t_1_0, DMA : 0)
    aie.flow(%t_0_3, DMA : 0, %t_1_1, DMA : 5)
    aie.packet_flow(3) { aie.packet_source<%t_2_3, DMA : 1> aie.packet_dest<%t_2_1, DMA : 4> }
    aie.packet_flow(13) { aie.packet_source<%t_2_5, Core : 0> aie.packet_dest<%t_2_2, DMA : 1> }
    aie.packet_flow(14) { aie.packet_source<%t_2_2, DMA : 1> aie.packet_dest<%t_2_1, DMA : 0> }
    aie.packet_flow(28) { aie.packet_source<%t_2_2, DMA : 1> aie.packet_source<%t_2_1, DMA : 3> aie.packet_dest<%t_2_0, DMA : 0> aie.packet_dest<%t_2_5, DMA : 0> }
    aie.packet_flow(17) { aie.packet_source<%t_2_1, DMA : 0> aie.packet_dest<%t_2_0, DMA : 0> }
  }
}

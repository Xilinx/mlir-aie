//===- packet_routing_priority_route.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s


// CHECK:    aie.tile(6, 2)
// CHECK:    aie.amsel<0> (0)
// CHECK:    aie.tile(6, 3)
// CHECK:    aie.amsel<0> (0)
// CHECK:    aie.tile(7, 2)
// CHECK:    aie.amsel<5> (3)
// CHECK:    aie.tile(7, 0)
// CHECK:    aie.amsel<5> (3)

//
// priority_route attribute overrides the downstream decision to assign arbiters and ports before other flows
//

module @aie_module  {
 aie.device(xcvc1902) {
  %t62 = aie.tile(6, 2)
  %t63 = aie.tile(6, 3)
  %t72 = aie.tile(7, 2)
  %t70 = aie.tile(7, 0)

  aie.packet_flow(0x1) {
    aie.packet_source<%t63, DMA : 0>
    aie.packet_dest<%t62, DMA : 1>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%t70, DMA : 0>
    aie.packet_dest<%t72, TileControl : 0>
  } {priority_route = true}
 }
}

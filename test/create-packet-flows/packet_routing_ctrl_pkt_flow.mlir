//===- packet_routing_ctrl_pkt_flow.mlir -----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
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
// ctrl_pkt_flow attribute overrides the downstream decision to use arbiters and ports reserved for contrl packets
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
    aie.packet_dest<%t72, Ctrl : 0>
  } {ctrl_pkt_flow = true}
 }
}

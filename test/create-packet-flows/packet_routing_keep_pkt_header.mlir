//===- packet_routing_keep_pkt_header.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1:    %[[VAL_0:.*]] = aie.tile(6, 2)
// CHECK1:    %[[VAL_1:.*]] = aie.tile(6, 3)
// CHECK1:    %[[VAL_2:.*]] = aie.tile(7, 2)
// CHECK1:    %[[VAL_3:.*]] = aie.tile(7, 3)
// CHECK1:    aie.packet_flow(1) {
// CHECK1:      aie.packet_source<%[[VAL_1:.*]], DMA : 0>
// CHECK1:      aie.packet_dest<%[[VAL_0:.*]], DMA : 1>
// CHECK1:    }
// CHECK1:    aie.packet_flow(2) {
// CHECK1:      aie.packet_source<%[[VAL_3:.*]], DMA : 0>
// CHECK1:      aie.packet_dest<%[[VAL_2:.*]], DMA : 1>
// CHECK1:    }

// CHECK2: "total_path_length": 2

//
// keep_pkt_header attribute overrides the downstream decision to drop the packet header
//

module @aie_module  {
 aie.device(xcvc1902) {
  %t62 = aie.tile(6, 2)
  %t63 = aie.tile(6, 3)
  %t72 = aie.tile(7, 2)
  %t73 = aie.tile(7, 3)

  aie.packet_flow(0x1) {
    aie.packet_source<%t63, DMA : 0>
    aie.packet_dest<%t62, DMA : 1>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%t73, DMA : 0>
    aie.packet_dest<%t72, DMA : 1>
  } {keep_pkt_header = true}
 }
}

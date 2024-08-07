//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// CHECK1: %[[T01:.*]] = aie.tile(0, 1)
// CHECK1: %[[T12:.*]] = aie.tile(1, 2)
// CHECK1: %[[T02:.*]] = aie.tile(0, 2)
// CHECK1: aie.packet_flow(16) {
// CHECK1:   aie.packet_source<%[[T01]], Core : 0>
// CHECK1:   aie.packet_dest<%[[T12]], Core : 0>
// CHECK1: }
// CHECK1: aie.packet_flow(16) {
// CHECK1:   aie.packet_source<%[[T01]], Core : 0>
// CHECK1:   aie.packet_dest<%[[T02]], DMA : 1>
// CHECK1: }
// CHECK1: aie.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)

// CHECK2: "total_path_length": 5

module {
  aie.device(xcvc1902) {
    %01 = aie.tile(0, 1)
    %12 = aie.tile(1, 2)
    %02 = aie.tile(0, 2)
    aie.flow(%01, DMA : 0, %12, Core : 1)
    aie.packet_flow(0x10) {
      aie.packet_source<%01, Core : 0>
      aie.packet_dest<%12, Core : 0>
      aie.packet_dest< %02, DMA : 1>
    }
  }
}

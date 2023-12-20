//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T01:.*]] = aie.tile(0, 1)
// CHECK: %[[T12:.*]] = aie.tile(1, 2)
// CHECK: aie.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)

module {
  aie.device(xcvc1902) {
    %01 = aie.tile(0, 1)
    %12 = aie.tile(1, 2)
    %02 = aie.tile(0, 2)
    aie.flow(%01, DMA : 0, %12, Core : 1)
    aie.packet_flow(0x10) {
      aie.packet_source < %01, Core : 0>
      aie.packet_dest < %12, Core : 0>
      aie.packet_dest < %02, DMA : 1>
    }
  }
}

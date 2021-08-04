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
// CHECK: %[[T01:.*]] = AIE.tile(0, 1)
// CHECK: %[[T12:.*]] = AIE.tile(1, 2)
// CHECK: AIE.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)

module {
  %01 = AIE.tile(0, 1)
  %12 = AIE.tile(1, 2)
  %02 = AIE.tile(0, 2)
  AIE.flow(%01, DMA : 0, %12, Core : 1)
  AIE.packet_flow(0x10) {
    AIE.packet_source < %01, Core : 0>
    AIE.packet_dest < %12, Core : 0>
    AIE.packet_dest < %02, DMA : 1>
  }
}

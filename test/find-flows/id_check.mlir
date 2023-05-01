//===- id_check.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt -aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: AIE.packet_flow(15) {
// CHECK:   AIE.packet_source<%[[T22]], DMA : 0>
// CHECK:   AIE.packet_dest<%[[T23]], DMA : 1>
// CHECK: }
module {
  AIE.device(xcvc1902) {
    %tile0 = AIE.tile(2, 3)
    %tile1 = AIE.tile(2, 2)

    %0 = AIE.switchbox(%tile0) {
      %16 = AIE.amsel<0> (0)
      %17 = AIE.masterset(DMA : 1, %16)
      AIE.packetrules(South : 0) {
        AIE.rule(7, 7, %16)
      }
    }
    %1 = AIE.switchbox(%tile1) {
      %18 = AIE.amsel<0> (0)
      %19 = AIE.masterset(North : 0, %18)
      AIE.packetrules(DMA : 0) {
        AIE.rule(15, 15, %18)
      }
    }
    AIE.wire(%0: Core, %tile0: Core)
    AIE.wire(%1: Core, %tile1: Core)
    AIE.wire(%0: DMA, %tile0: DMA)
    AIE.wire(%1: DMA, %tile1: DMA)
    AIE.wire(%0: South, %1: North)
  }
}

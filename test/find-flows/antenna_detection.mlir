//===- find_flows.mlir -----------------------------------------*- MLIR -*-===//
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
// CHECK: AIE.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
// CHECK: AIE.flow(%[[T22]], Core : 0, %[[T22]], Core : 0)
// CHECK: AIE.flow(%[[T22]], Core : 1, %[[T23]], Core : 1)
// CHECK: AIE.packet_flow(0) {
// CHECK:   AIE.packet_source<%[[T22]], DMA : 0>
// CHECK:   AIE.packet_dest<%[[T23]], DMA : 1>
// CHECK: }
// CHECK: AIE.flow(%2, Core : 0, %9, DMA : 0)
module {
  %tile0 = AIE.tile(2, 3)
  %tile1 = AIE.tile(2, 2)
  %tile21 = AIE.tile(2, 1)
  %tile20 = AIE.tile(2, 0)

  %0 = AIE.switchbox(%tile0) {
    AIE.connect<Core :0, South:1>
    AIE.connect<South:1, Core :1>
    AIE.connect<South:1, North:2> //endpoint antenna
    AIE.connect<South:3, Core :0> //dangling island antenna 
    %16 = AIE.amsel<0> (0)
    %17 = AIE.masterset(DMA : 1, %16)
    AIE.packetrules(South : 0) {
      AIE.rule(31, 0, %16)
    }
  }
  %1 = AIE.switchbox(%tile1) {
    AIE.connect<North:1, Core :1>
    AIE.connect<Core :1, North:1>
    AIE.connect<Core :0, Core :0>
    AIE.connect<East :1, North :3> //dangling island antenna 
    %18 = AIE.amsel<0> (0)
    %19 = AIE.masterset(North : 0, %18)
    // packet antennas
    %20 = AIE.amsel<0> (1)
    %21 = AIE.masterset(East : 1, %20)
    // packet antennas
    AIE.packetrules(DMA : 0) {
      AIE.rule(31, 0, %18)
      AIE.rule(31, 0, %20)
    }
    // dangling antenna
    %22 = AIE.amsel<1> (2)
    %23 = AIE.masterset(East : 2, %22)
    AIE.packetrules(West : 0) {
      AIE.rule(31, 0, %22)
    }
  }

  %s21 = AIE.switchbox(%tile21)  {
    AIE.connect<Core : 0, South : 0>
  }
  %s20 = AIE.switchbox(%tile20)  {
    AIE.connect<North : 0, South : 2>
  }
  %mux = AIE.shimmux(%tile20)  {
    AIE.connect<North : 2, DMA : 0>
    // AIE.connect<North : 2, East : 1> //endpoint antenna
    // AIE.connect<West : 3, East : 3> //dangling island antenna 
  }
  %dma = AIE.shimDMA(%tile20)  {
    AIE.end
  }

  AIE.wire(%0: Core, %tile0: Core)
  AIE.wire(%1: Core, %tile1: Core)
  AIE.wire(%0: DMA, %tile0: DMA)
  AIE.wire(%1: DMA, %tile1: DMA)
  AIE.wire(%0: South, %1: North)
  
  AIE.wire(%s21 : Core, %tile21 : Core)
  AIE.wire(%s21 : South, %s20 : North)
  AIE.wire(%s20 : South, %mux : North)
  AIE.wire(%mux : DMA, %dma : DMA)
}
  
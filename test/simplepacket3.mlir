//===- simplepacket3.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// CHECK-LABEL: module {
// CHECK:       }

module {
  %0 = aie.tile(0, 1)
  %1 = aie.tile(1, 2)
  %2 = aie.tile(0, 2)
  %3 = aie.tile(1, 1)
  %4 = aie.switchbox(%0) {
    aie.connect<DMA : 0, North : 0>
  }
  %5 = aie.switchbox(%2) {
    // aie.connect<South : 0, East : 0>
    %a1_0 = aie.amsel<1>(0)
    %m1 = aie.masterset(East : 0, %a1_0 )
    aie.packet_rules(South : 0) {
      aie.rule(0x1F, 0x10, %a1_0)
    }
  }
  %6 = aie.switchbox(%3) {
    aie.connect<North : 0, Core : 1>
  }
  %7 = aie.switchbox(%1) {
    %a1_0 = aie.amsel<1>(0)
    %m1 = aie.masterset(South : 0, %a1_0 )
    aie.packet_rules(West : 0) {
      aie.rule(0x10, 0x0, %a1_0)
    }
  }
  %8 = aie.shim_switchbox(0) {
  }
  %9 = aie.shim_switchbox(1) {
  }
  %10 = aie.plio(0)
  %11 = aie.plio(1)
  aie.wire(%0 : Core, %4 : Core)
  aie.wire(%0 : DMA, %4 : DMA)
  aie.wire(%8 : North, %4 : South)
  aie.wire(%10 : North, %8 : South)
  aie.wire(%2 : Core, %5 : Core)
  aie.wire(%2 : DMA, %5 : DMA)
  aie.wire(%4 : North, %5 : South)
  aie.wire(%3 : Core, %6 : Core)
  aie.wire(%3 : DMA, %6 : DMA)
  aie.wire(%4 : East, %6 : West)
  aie.wire(%9 : North, %6 : South)
  aie.wire(%8 : East, %9 : West)
  aie.wire(%11 : North, %9 : South)
  aie.wire(%1 : Core, %7 : Core)
  aie.wire(%1 : DMA, %7 : DMA)
  aie.wire(%5 : East, %7 : West)
  aie.wire(%6 : North, %7 : South)

}

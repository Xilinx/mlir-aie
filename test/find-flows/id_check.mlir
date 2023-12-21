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
// CHECK: %[[T23:.*]] = aie.tile(2, 3)
// CHECK: %[[T22:.*]] = aie.tile(2, 2)
// CHECK: aie.packet_flow(15) {
// CHECK:   aie.packet_source<%[[T22]], DMA : 0>
// CHECK:   aie.packet_dest<%[[T23]], DMA : 1>
// CHECK: }
module {
  aie.device(xcvc1902) {
    %tile0 = aie.tile(2, 3)
    %tile1 = aie.tile(2, 2)

    %0 = aie.switchbox(%tile0) {
      %16 = aie.amsel<0> (0)
      %17 = aie.masterset(DMA : 1, %16)
      aie.packet_rules(South : 0) {
        aie.rule(7, 7, %16)
      }
    }
    %1 = aie.switchbox(%tile1) {
      %18 = aie.amsel<0> (0)
      %19 = aie.masterset(North : 0, %18)
      aie.packet_rules(DMA : 0) {
        aie.rule(15, 15, %18)
      }
    }
    aie.wire(%0: Core, %tile0: Core)
    aie.wire(%1: Core, %tile1: Core)
    aie.wire(%0: DMA, %tile0: DMA)
    aie.wire(%1: DMA, %tile1: DMA)
    aie.wire(%0: South, %1: North)
  }
}

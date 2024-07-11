//===- memtile_routing_constraints.mlir ------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s

// CHECK: %[[T24:.*]] = aie.tile(2, 4)
// CHECK: %[[T23:.*]] = aie.tile(2, 3)
// CHECK: %[[T22:.*]] = aie.tile(2, 2)
// CHECK: %[[T21:.*]] = aie.tile(2, 1)
// CHECK: %[[T20:.*]] = aie.tile(2, 0)
// CHECK: aie.switchbox(%[[T21]]) {
// CHECK:   aie.connect<North : 0, DMA : 0>
// CHECK:   aie.connect<North : 1, South : 1>
// CHECK: }
// CHECK: aie.switchbox(%[[T22]]) {
// CHECK:   aie.connect<DMA : 0, South : 0>
// CHECK:   aie.connect<North : 1, South : 1>
// CHECK: }
// CHECK: aie.switchbox(%[[T20]]) {
// CHECK:   aie.connect<North : 1, South : 2>
// CHECK: }
// CHECK: aie.switchbox(%[[T23]]) {
// CHECK:   aie.connect<DMA : 0, South : 1>
// CHECK: }

module {
    aie.device(xcve2802) {
        %t24 = aie.tile(2, 4)
        %t23 = aie.tile(2, 3)
        %t22 = aie.tile(2, 2)
        %t21 = aie.tile(2, 1)
        %t20 = aie.tile(2, 0)

        aie.flow(%t22, DMA : 0, %t21, DMA : 0)
        aie.flow(%t23, DMA : 0, %t20, DMA : 0)
    }
}
